from agent.base_agent import BaseAgent
import torch


class Agent2x(BaseAgent):
    def __init__(self, config, net):
        super(Agent2x, self).__init__(config, net)

    def forward(self, data):
        input1 = data['input1'].to(self.device)
        input2 = data['input2'].to(self.device)
        input12 = data['input12'].to(self.device)
        input21 = data['input21'].to(self.device)
        target1 = data['target1'].to(self.device)
        target2 = data['target2'].to(self.device)
        target12 = data['target12'].to(self.device)
        target21 = data['target21'].to(self.device)

        losses = {}

        if self.use_triplet:
            outputs, motionvecs, staticvecs = self.net.cross_with_triplet(input1, input2, input12, input21)
            losses['m_tpl1'] = self.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
            losses['m_tpl2'] = self.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
            losses['b_tpl1'] = self.triplet_weight * self.tripletloss(staticvecs[2], staticvecs[0], staticvecs[1])
            losses['b_tpl2'] = self.triplet_weight * self.tripletloss(staticvecs[3], staticvecs[1], staticvecs[0])
        else:
            outputs = self.net.cross(input1, input2)

        # update loss metric
        losses['v1'] = self.mse(outputs[0], target1)
        losses['v2'] = self.mse(outputs[1], target2)
        losses['v12'] = self.mse(outputs[2], target12)
        losses['v21'] = self.mse(outputs[3], target21)

        outputs = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output12": outputs[2],
            "output21": outputs[3],
        }
        return outputs, losses


class Agent3x(BaseAgent):
    def __init__(self, config, net):
        super(Agent3x, self).__init__(config, net)

    def forward(self, data):
        input1 = data['input1'].to(self.device)
        input2 = data['input2'].to(self.device)
        target1 = data['target111'].to(self.device)
        target2 = data['target222'].to(self.device)
        target121 = data['target121'].to(self.device)
        target112 = data['target112'].to(self.device)
        target122 = data['target122'].to(self.device)
        target212 = data['target212'].to(self.device)
        target221 = data['target221'].to(self.device)
        target211 = data['target211'].to(self.device)

        out1, out2, out121, out112, out122, out212, out221, out211 = self.net.cross(input1, input2)

        # update loss metric
        losses = {}
        losses['v111'] = self.mse(out1, target1)
        losses['v222'] = self.mse(out2, target2)
        losses['v121'] = self.mse(out121, target121)
        losses['v112'] = self.mse(out112, target112)
        losses['v122'] = self.mse(out122, target122)
        losses['v212'] = self.mse(out212, target212)
        losses['v221'] = self.mse(out221, target221)
        losses['v211'] = self.mse(out211, target211)

        # TODO : reorganize triplet loss

        outputs = {
            "output111": out1,
            "output222": out2,
            "output121": out121,
            "output112": out112,
            "output122": out122,
            "output212": out212,
            "output221": out221,
            "output211": out211,
        }

        return outputs, losses
