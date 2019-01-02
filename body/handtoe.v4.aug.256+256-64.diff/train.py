import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataset import get_dataloaders, MEAN_POSE, STD_POSE
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

        self.foot_idx = config.foot_idx
        self.fv_weight = config.fv_weight

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

    def footvel_loss(self, output, target):
        # toe base: 12, 16
        gv_out = output[:, -2:, 1:] - output[:, -2:, :-1]
        gv_tgt = target[:, -2:, 1:] - target[:, -2:, :-1]
        lv_out = output[:, self.foot_idx, 1:] - output[:, self.foot_idx, :-1]
        lv_tgt = target[:, self.foot_idx, 1:] - target[:, self.foot_idx, :-1]
        fv_out = lv_out + gv_out.repeat(1, 2, 1)
        fv_tgt = lv_tgt + gv_tgt.repeat(1, 2, 1)
        return F.mse_loss(fv_out, fv_tgt)

    def train_func(self, data):
        # get data
        self.net.train()
        input1 = data['input1'].to(config.device)
        input2 = data['input2'].to(config.device)
        target1 = data['target1'].to(config.device)
        target2 = data['target2'].to(config.device)
        target12 = data['target12'].to(config.device)
        target21 = data['target21'].to(config.device)

        # pass through the model
        output1, output2, output12, output21 = self.net.cross(input1, input2)

        # update loss metric
        losses = {}
        losses['v1'] = F.mse_loss(output1, target1)
        losses['v2'] = F.mse_loss(output2, target2)
        losses['v12'] = F.mse_loss(output12, target12)
        losses['v21'] = F.mse_loss(output21, target21)
        losses['footvel'] = self.fv_weight * (self.footvel_loss(output1, target1)
                             + self.footvel_loss(output2, target2)
                             + self.footvel_loss(output12, target12)
                             + self.footvel_loss(output21, target21)) / 4.0

        outputs = {
            "output1": output1,
            "output2": output2,
            "output12": output12,
            "output21": output21,
        }

        return outputs, losses

    def val_func(self, data):
        # get data
        self.net.eval()
        input1 = data['input1'].to(config.device)
        input2 = data['input2'].to(config.device)
        target1 = data['target1'].to(config.device)
        target2 = data['target2'].to(config.device)
        target12 = data['target12'].to(config.device)
        target21 = data['target21'].to(config.device)

        # pass through the model
        with torch.no_grad():
            output1, output2, output12, output21 = self.net.cross(input1, input2)

            # update loss metric
            losses = {}
            losses['v1'] = F.mse_loss(output1, target1)
            losses['v2'] = F.mse_loss(output2, target2)
            losses['v12'] = F.mse_loss(output12, target12)
            losses['v21'] = F.mse_loss(output21, target21)
            losses['footvel'] = self.fv_weight * (self.footvel_loss(output1, target1)
                                 + self.footvel_loss(output2, target2)
                                 + self.footvel_loss(output12, target12)
                                 + self.footvel_loss(output21, target21)) / 4.0

        outputs = {
            "output1": output1,
            "output2": output2,
            "output12": output12,
            "output21": output21,
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
    #logger = WorklogLogger(os.path.join(config.log_dir, 'log.txt'))

    # create tensorboard writer
    train_tb = SummaryWriter(os.path.join(sess.log_dir, 'train.events'))
    val_tb = SummaryWriter(os.path.join(sess.log_dir, 'val.events'))

    # create dataloader
    train_loader = get_dataloaders('train', batch_size=config.batch_size,
                                   shuffle=True)
    val_loader = get_dataloaders('validation', batch_size=config.batch_size)
    #train_cluster_data = train_loader.dataset.get_cluster_data()
    #val_cluster_data = val_loader.dataset.get_cluster_data()
    val_loader = cycle(val_loader)

    # set criterion and AverageMeter to calc and monitor loss

    # set optimizer
    optimizer = optim.Adam(sess.net.parameters(), config.lr)

    # set learning rate scheduler
    scheduler = ExponentialLR(optimizer, 0.99)

    # prepare for clustering
    '''
    cluster_dir = os.path.join(sess.log_dir, 'cluster')
    ensure_dir(cluster_dir)
    ensure_dir(os.path.join(cluster_dir, 'train.motion'))
    ensure_dir(os.path.join(cluster_dir, 'train.body'))
    ensure_dir(os.path.join(cluster_dir, 'validation.motion'))
    ensure_dir(os.path.join(cluster_dir, 'validation.body'))
    '''
    # start session
    clock = sess.clock
    net = sess.net
    sess.save_checkpoint('start.pth.tar')

    # start training
    net.train()
    for e in range(config.nr_epochs):
        # evaluate clustering
        '''
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
            else:
                val_tb.add_image(k, v, global_step=e)
        '''
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # pass through net and get loss
            outputs, losses = sess.train_func(data)
            losses_values = {k:v.item() for k, v in losses.items()}

            # update loss metric
            loss = losses['v1'] + losses['v2'] + losses['v12'] + losses['v21']
                   #+ losses['footvel']

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''
            # write to log
            logger.put_line(
                "Train: EPOCH[{}][{}/{}]: ".format(e, b, len(train_loader)) +
                " ".join([
                    "{}={:.2g}".format(k, v) for k, v in losses_values.items()
                ])
            )
            '''

            # update tensorboard
            for k, v in losses_values.items():
                train_tb.add_scalar(k, v, clock.step)

            # visualize
            if clock.step % config.visualize_frequency == 0:
                skeleton_maps = visulize_motion_in_training(data['input1'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('input_1', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(data['input2'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('input_2', torch.Tensor(skeleton_maps), clock.step)

                skeleton_maps = visulize_motion_in_training(outputs['output12'].detach().cpu().numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('output_12', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(outputs['output21'].detach().cpu().numpy()[0],
                                                            MEAN_POSE, STD_POSE)
                train_tb.add_image('output_21', torch.Tensor(skeleton_maps), clock.step)

                skeleton_maps = visulize_motion_in_training(data['target12'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('target_12', torch.Tensor(skeleton_maps), clock.step)
                skeleton_maps = visulize_motion_in_training(data['target21'].numpy()[0], MEAN_POSE, STD_POSE)
                train_tb.add_image('target_21', torch.Tensor(skeleton_maps), clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
            pbar.set_postfix(OrderedDict(losses_values))

            # validation
            if clock.step % config.val_frequency == 0:
                data = next(val_loader)

                outputs, losses = sess.val_func(data)
                losses_values = {k:v.item() for k, v in losses.items()}
                '''
                logger.put_line(
                    "Val: EPOCH[{}][{}/{}]: ".format(e, b, len(train_loader)) +
                    " ".join([
                        "{}={:.2g}".format(k, v) for k, v in losses_values.items()
                    ])
                )
                '''
                for k, v in losses_values.items():
                    val_tb.add_scalar(k, v, clock.step)

                if clock.step % config.visualize_frequency == 0:
                    skeleton_maps = visulize_motion_in_training(data['input1'].numpy()[0], MEAN_POSE, STD_POSE)
                    val_tb.add_image('input_1', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(data['input2'].numpy()[0], MEAN_POSE, STD_POSE)
                    val_tb.add_image('input_2', torch.Tensor(skeleton_maps), clock.step)

                    skeleton_maps = visulize_motion_in_training(outputs['output12'].detach().cpu().numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('output_12', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(outputs['output21'].detach().cpu().numpy()[0],
                                                                MEAN_POSE, STD_POSE)
                    val_tb.add_image('output_21', torch.Tensor(skeleton_maps), clock.step)

                    skeleton_maps = visulize_motion_in_training(data['target12'].numpy()[0], MEAN_POSE, STD_POSE)
                    val_tb.add_image('target_12', torch.Tensor(skeleton_maps), clock.step)
                    skeleton_maps = visulize_motion_in_training(data['target21'].numpy()[0], MEAN_POSE, STD_POSE)
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
