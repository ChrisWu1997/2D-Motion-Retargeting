from functional.utils import TrainClock
import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod


class BaseAgent(object):
    def __init__(self, config, net):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.clock = TrainClock()
        self.device = config.device

        self.use_triplet = config.use_triplet
        self.use_footvel_loss = config.use_footvel_loss

        # set loss function
        self.mse = nn.MSELoss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.triplet_weight = config.triplet_weight
        self.foot_idx = config.foot_idx
        self.footvel_loss_weight = config.footvel_loss_weight

        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)

    def save_network(self, name=None):
        if name is None:
            save_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, name)
        torch.save(self.net.cpu().state_dict(), save_path)
        self.net.to(self.device)

    def load_network(self, epoch):
        load_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(epoch))
        state_dict = torch.load(load_path)
        self.net.load_state_dict(state_dict)

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data):
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)

        return outputs, losses

    def val_func(self, data):
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        return outputs, losses
