import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
from common import config
from model import AutoEncoder
from utils import TrainClock, ensure_dir, WorklogLogger, cycle
from dataset import get_dataloaders, MEAN_POSE, STD_POSE, STD_POSE
from visulization import visulize_motion_in_training, cluster_in_training

torch.backends.cudnn.benchmark = True


class Session:

    def __init__(self, config, net=None):
        self.log_dir = config.log_dir
        ensure_dir(self.log_dir)
        self.model_dir = config.model_dir
        ensure_dir(self.model_dir)
        self.net = net
        self.best_val_acc = 0.0
        self.clock = TrainClock()

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            'net': self.net,
            'best_val_acc': self.best_val_acc,
            'clock': self.clock.make_checkpoint(),
        }
        torch.save(tmp, ckp_path)

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.net = checkpoint['net']
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.best_val_acc = checkpoint['best_val_acc']

    def set_criterion_dict(self):
        self.criterion = nn.MSELoss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)

    def weighted_mse(self, output, target):
        return torch.mean((output - target) ** 2 * self.weight[:output.shape[0]])

    def set_loss_weight(self, weight, device):
        self.weight = torch.tensor(weight, requires_grad=False, dtype=torch.float).to(device)

    def train_func(self, data):
        # get data
        self.net.train()
        input1 = data['input1'].to(config.device)
        input2 = data['input2'].to(config.device)
        input12 = data['input12'].to(config.device)
        input21 = data['input21'].to(config.device)
        target1 = data['target1'].to(config.device)
        target2 = data['target2'].to(config.device)
        target12 = data['target12'].to(config.device)
        target21 = data['target21'].to(config.device)

        # pass through the model
        outputs, motionvecs, viewvecs = self.net.cross_with_triplet(input1, input2, input12, input21)

        # update loss metric
        losses = {}
        losses['v1'] = self.weighted_mse(outputs[0], target1)
        losses['v2'] = self.weighted_mse(outputs[1], target2)
        losses['v12'] = self.weighted_mse(outputs[2], target12)
        losses['v21'] = self.weighted_mse(outputs[3], target21)

        # triplet loss
        losses['m_tpl1'] = config.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
        losses['m_tpl2'] = config.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
        losses['v_tpl1'] = config.triplet_weight * self.tripletloss(viewvecs[2], viewvecs[0], viewvecs[1])
        losses['v_tpl2'] = config.triplet_weight * self.tripletloss(viewvecs[3], viewvecs[1], viewvecs[0])

        outputs = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output12": outputs[2],
            "output21": outputs[3],
        }

        return outputs, losses

    def val_func(self, data):
        # get data
        self.net.eval()
        input1 = data['input1'].to(config.device)
        input2 = data['input2'].to(config.device)
        input12 = data['input12'].to(config.device)
        input21 = data['input21'].to(config.device)
        target1 = data['target1'].to(config.device)
        target2 = data['target2'].to(config.device)
        target12 = data['target12'].to(config.device)
        target21 = data['target21'].to(config.device)

        # pass through the model
        with torch.no_grad():
            # pass through the model
            outputs, motionvecs, viewvecs = self.net.cross_with_triplet(input1, input2, input12, input21)

            # update loss metric
            losses = {}
            losses['v1'] = self.weighted_mse(outputs[0], target1)
            losses['v2'] = self.weighted_mse(outputs[1], target2)
            losses['v12'] = self.weighted_mse(outputs[2], target12)
            losses['v21'] = self.weighted_mse(outputs[3], target21)

            # triplet loss
            losses['m_tpl1'] = config.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
            losses['m_tpl2'] = config.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
            losses['v_tpl1'] = config.triplet_weight * self.tripletloss(viewvecs[2], viewvecs[0], viewvecs[1])
            losses['v_tpl2'] = config.triplet_weight * self.tripletloss(viewvecs[3], viewvecs[1], viewvecs[0])

        outputs = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output12": outputs[2],
            "output21": outputs[3],
        }

        return outputs, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('train_log'):
        os.symlink(config.exp_dir, 'train_log')

    net = AutoEncoder(config.mot_en_channels,
                      config.body_en_channels,
                      config.de_channels,
                      config.len_frames)
    print(net)
    net = net.cuda()

    # create session
    sess = Session(config, net=net)
    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    # create logger
    logger = WorklogLogger(os.path.join(config.log_dir, 'log.txt'))

    # create tensorboard writer
    train_tb = SummaryWriter(os.path.join(sess.log_dir, 'train.events'))
    val_tb = SummaryWriter(os.path.join(sess.log_dir, 'val.events'))

    # create dataloader
    train_loader = get_dataloaders('train', batch_size=config.batch_size,
                                   shuffle=True)
    val_loader = get_dataloaders('validation', batch_size=config.batch_size)
    train_cluster_data = train_loader.dataset.get_cluster_data()
    val_cluster_data = val_loader.dataset.get_cluster_data()
    val_loader = cycle(val_loader)

    # set criterion and AverageMeter to calc and monitor loss
    sess.set_criterion_dict()
    loss_weight = np.ones((config.batch_size, config.len_joints, config.len_frames))
    loss_weight[:, [20, 21, 26, 27], :] = config.foot_weight
    sess.set_loss_weight(loss_weight, device=config.device)

    # set optimizer
    optimizer = optim.Adam(sess.net.parameters(), config.lr)

    # set learning rate scheduler
    #scheduler = ExponentialLR(optimizer, 0.99)
    scheduler = StepLR(optimizer, 200)

    # prepare for clustering
    cluster_dir = os.path.join(sess.log_dir, 'cluster')
    ensure_dir(cluster_dir)
    ensure_dir(os.path.join(cluster_dir, 'train.motion'))
    ensure_dir(os.path.join(cluster_dir, 'train.view'))
    ensure_dir(os.path.join(cluster_dir, 'validation.motion'))
    ensure_dir(os.path.join(cluster_dir, 'validation.view'))

    # start session
    clock = sess.clock
    net = sess.net
    sess.save_checkpoint('start.pth.tar')

    # start training
    net.train()
    for e in range(config.nr_epochs):
        # evaluate clustering
        net.eval()
        clusters = cluster_in_training(net, train_cluster_data, 'train', config.device, cluster_dir, e)
        for k, v in clusters.items():
            if 'score' in k:
                train_tb.add_scalar(k, v, global_step=e)
            elif v is not None:
                train_tb.add_image(k, v, global_step=e)

        clusters = cluster_in_training(net, val_cluster_data, 'validation', config.device, cluster_dir, e)
        for k, v in clusters.items():
            if 'score' in k:
                val_tb.add_scalar(k, v, global_step=e)
            elif v is not None:
                val_tb.add_image(k, v, global_step=e)

        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # pass through net and get loss
            outputs, losses = sess.train_func(data)
            losses_values = {k:v.item() for k, v in losses.items()}

            # update loss metric
            loss = losses['v1'] + losses['v2'] + losses['v12'] + losses['v21'] \
                + losses['m_tpl1'] + losses['m_tpl2'] + losses['v_tpl1'] + losses['v_tpl2']

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to log
            logger.put_line(
                "Train: EPOCH[{}][{}/{}]: ".format(e, b, len(train_loader)) +
                " ".join([
                    "{}={:.2g}".format(k, v) for k, v in losses_values.items()
                ])
            )

            # update tensorboard
            for k, v in losses_values.items():
                train_tb.add_scalar(k, v, clock.step)

            # visualize
            if clock.step % config.visualize_frequency == 0:
                skeleton_maps = visulize_motion_in_training(data['input1'].numpy()[0], data['center1'].numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('input_1', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(data['input2'].numpy()[0], data['center2'].numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('input_2', torch.Tensor(skeleton_maps), clock.step)

                skeleton_maps = visulize_motion_in_training(outputs['output12'].detach().cpu().numpy()[0],
                                                data['center1'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('output_12', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(outputs['output21'].detach().cpu().numpy()[0],
                                                data['center2'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('output_21', torch.Tensor(skeleton_maps), clock.step)

                skeleton_maps = visulize_motion_in_training(data['target12'].numpy()[0], data['center12'].numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('target_12', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(data['target21'].numpy()[0], data['center21'].numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('target_21', torch.Tensor(skeleton_maps), clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
            pbar.set_postfix(OrderedDict(losses_values))

            # validation
            if clock.step % config.val_frequency == 0:
                data = next(val_loader)

                outputs, losses = sess.val_func(data)
                losses_values = {k:v.item() for k, v in losses.items()}

                logger.put_line(
                    "Val: EPOCH[{}][{}/{}]: ".format(e, b, len(train_loader)) +
                    " ".join([
                        "{}={:.2g}".format(k, v) for k, v in losses_values.items()
                    ])
                )

                for k, v in losses_values.items():
                    val_tb.add_scalar(k, v, clock.step)

                if clock.step % config.visualize_frequency == 0:
                    skeleton_maps = visulize_motion_in_training(data['input1'].numpy()[0], data['center1'].numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('input_1', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(data['input2'].numpy()[0], data['center2'].numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('input_2', torch.Tensor(skeleton_maps), clock.step)

                    skeleton_maps = visulize_motion_in_training(outputs['output12'].detach().cpu().numpy()[0],
                                                    data['center1'].numpy()[0], MEAN_POSE, STD_POSE)
                    val_tb.add_image('output_12', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(outputs['output21'].detach().cpu().numpy()[0],
                                                    data['center2'].numpy()[0], MEAN_POSE, STD_POSE)
                    val_tb.add_image('output_21', torch.Tensor(skeleton_maps), clock.step)

                    skeleton_maps = visulize_motion_in_training(data['target12'].numpy()[0], data['center12'].numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('target_12', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(data['target21'].numpy()[0], data['center21'].numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('target_21', torch.Tensor(skeleton_maps), clock.step)

            clock.tick()

        train_tb.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], clock.epoch)
        scheduler.step(clock.epoch)

        if clock.epoch % config.save_frequency == 0:
            sess.save_checkpoint('epoch{}.pth.tar'.format(clock.epoch))
        sess.save_checkpoint('latest.pth.tar')

        clock.tock()


if __name__ == '__main__':
    main()
