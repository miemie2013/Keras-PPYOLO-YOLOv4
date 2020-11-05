#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
import tensorflow as tf
import copy
from collections import OrderedDict



class YOLO(object):
    def __init__(self, backbone, head, ema_decay=0.9998):
        super(YOLO, self).__init__()
        self.backbone = backbone
        self.head = head
        self.ema_decay = ema_decay
        self.ema_state_dict = OrderedDict()
        self.current_state_dict = OrderedDict()

    def get_outputs(self, x):
        body_feats = self.backbone(x)
        outputs = self.head._get_outputs(body_feats)
        return outputs

    def get_prediction(self, outputs, im_size):
        preds = self.head.get_prediction(outputs, im_size)
        return preds

    def get_loss(self, args):
        output0 = args[0]
        output1 = args[1]
        output2 = args[2]
        gt_box = args[3]
        target0 = args[4]
        target1 = args[5]
        target2 = args[6]
        outputs = [output0, output1, output2]
        targets = [target0, target1, target2]
        loss = self.head.get_loss(outputs, gt_box, None, None, targets)
        return loss

    def init_ema_state_dict(self):
        pass

    def update_ema_state_dict(self, thres_steps):
        pass

    def apply_ema_state_dict(self):
        pass

    def restore_current_state_dict(self):
        pass




