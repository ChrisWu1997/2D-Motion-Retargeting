from dataset import get_dataloader
from common import config
from model import get_autoencoder
from functional.utils import cycle
from agent import get_training_agent
from functional.visualization import visulize_motion_in_training
import torch
import os
from collections import OrderedDict
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use')
    # parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--disable_triplet', action='store_true', default=False, help="disable triplet loss")
    parser.add_argument('--use_footvel_loss', action='store_true', default=False, help="use use footvel loss")
    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()

    config.initialize(args)

    net = get_autoencoder(config)
    print(net)
    net = net.to(config.device)

    # create tensorboard writer
    train_tb = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
    val_tb = SummaryWriter(os.path.join(config.log_dir, 'val.events'))

    # create dataloader
    train_loader = get_dataloader('train', config, config.batch_size, config.num_workers)
    mean_pose, std_pose = train_loader.dataset.mean_pose, train_loader.dataset.std_pose
    val_loader = get_dataloader('test', config, config.batch_size, config.num_workers)
    val_loader = cycle(val_loader)

    # create training agent
    tr_agent = get_training_agent(config, net)
    clock = tr_agent.clock

    # start training
    for e in range(config.nr_epochs):

        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            losses_values = {k:v.item() for k, v in losses.items()}

            # record loss to tensorboard
            for k, v in losses_values.items():
                train_tb.add_scalar(k, v, clock.step)

            # visualize
            if args.vis and clock.step % config.visualize_frequency == 0:
                imgs = visulize_motion_in_training(outputs, mean_pose, std_pose)
                for k, img in imgs.items():
                    train_tb.add_image(k, torch.from_numpy(img), clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
            pbar.set_postfix(OrderedDict({"loss": sum(losses_values.values())}))

            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(val_loader)

                outputs, losses = tr_agent.val_func(data)

                losses_values = {k: v.item() for k, v in losses.items()}

                for k, v in losses_values.items():
                    val_tb.add_scalar(k, v, clock.step)

                if args.vis and clock.step % config.visualize_frequency == 0:
                    imgs = visulize_motion_in_training(outputs, mean_pose, std_pose)
                    for k, img in imgs.items():
                        val_tb.add_image(k, torch.from_numpy(img), clock.step)

            clock.tick()

        train_tb.add_scalar('learning_rate', tr_agent.optimizer.param_groups[-1]['lr'], clock.epoch)
        tr_agent.update_learning_rate()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_network()
        tr_agent.save_network('latest.pth.tar')

        clock.tock()


if __name__ == '__main__':
    main()
