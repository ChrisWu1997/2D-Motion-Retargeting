import numpy as np
from copy import copy
from dataset.base_dataset import _MixamoDatasetBase


class MixamoDatasetForSkeleton(_MixamoDatasetBase):
    def __init__(self, phase, config):
        super(MixamoDatasetForSkeleton, self).__init__(phase, config)

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2, replace=False)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]

        if self.aug:
            param1 = self.gen_aug_param(rotate=True)
            param2 = self.gen_aug_param(rotate=True)
            param12 = copy(param1)
            param21 = copy(param2)
            param12['ratio'] = param2['ratio']
            param21['ratio'] = param1['ratio']
        else:
            param1 = param2 = param12 = param21 = None

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12 = self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        input1 = self.preprocessing(item1, param=param1)
        input2 = self.preprocessing(item2, param=param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        input12 = self.preprocessing(item12, param=param12)
        input21 = self.preprocessing(item21, param=param21)
        target12 = input12.detach().clone()
        target21 = input21.detach().clone()

        return {"input1": input1, "target1": target1,
                "input2": input2, "target2": target2,
                "input12": input12, "target12": target12,
                "input21": input21, "target21": target21,
                "mot1": mot1, "mot2": mot2,
                "char1": char1, "char2": char2}


class MixamoDatasetForView(_MixamoDatasetBase):
    def __init__(self, phase, config):
        super(MixamoDatasetForView, self).__init__(phase, config)

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]
        # select two views
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2, replace=False)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12= self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        if self.aug:
            param1 = self.gen_aug_param(rotate=False)   # FIXME: [np.random.uniform(0.5, 1.5)]
            param2 = self.gen_aug_param(rotate=False)
            param12 = param2
            param21 = param1
        else:
            param1 = param2 = param12 = param21 = None

        input1 = self.preprocessing(item1, view1, param1)
        input2 = self.preprocessing(item2, view2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        input12 = self.preprocessing(item12, view2, param12)
        input21 = self.preprocessing(item21, view1, param21)
        target12 = input12.detach().clone()
        target21 = input21.detach().clone()

        return {"input1": input1, "target1": target1,
                "input2": input2, "target2": target2,
                "input12": input12, "target12": target12,
                "input21": input21, "target21": target21,
                "mot1": mot1, "mot2": mot2,
                "view1": view1, "view2": view2,
                "char1": char1, "char2": char2}


class MixamoDatasetForFull(_MixamoDatasetBase):
    def __init__(self, phase, config):
        super(MixamoDatasetForFull, self).__init__(phase, config)

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2, replace=False)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]
        # select two views
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2, replace=False)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12= self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        if self.aug:
            param1 = self.gen_aug_param(rotate=False)   # FIXME: [np.random.uniform(0.5, 1.5)]
            param2 = self.gen_aug_param(rotate=False)
        else:
            param1 = param2 = None

        input1 = self.preprocessing(item1, view1, param1)
        input2 = self.preprocessing(item2, view2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        target112 = self.preprocessing(item1, view2, param1)
        target121 = self.preprocessing(item12, view1, param2)
        target122 = self.preprocessing(item12, view2, param2)
        target221 = self.preprocessing(item2, view1, param2)
        target212 = self.preprocessing(item21, view2, param1)
        target211 = self.preprocessing(item21, view1, param1)

        return {"input1": input1, "target111": target1,
                "input2": input2, "target222": target2,
                "target112": target112,
                "target121": target121,
                "target122": target122,
                "target221": target221,
                "target212": target212,
                "target211": target211,
                "mot1": mot1, "mot2": mot2,
                "view1": view1, "view2": view2,
                "char1": char1, "char2": char2}
