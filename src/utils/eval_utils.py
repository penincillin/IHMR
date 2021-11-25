import cv2
import numpy as np
import os.path as osp
import sys
import time
import ry_utils


class ResultStat(object):
    def __init__(self):
        self.result_info = [
            ('mpjpe_3d', 'small', 1),
            ('inter_mpjpe_3d', 'small', 1),
            ('collision_ave', 'small', 1),
            ('collision_max', 'small', 1),
        ]
        # save all results
        self.all_results = dict()
        self.best_results = dict()
        self.get_best_results = dict()
        for metric, result_type, scale_ratio in self.result_info:
            assert result_type in ['large', 'small']
            self.all_results[metric] = (result_type, scale_ratio, list())
            self.best_results[metric] = None
            self.get_best_results[metric] = False
    

    def update(self, metric, epoch, value):
        # add to all results
        self.all_results[metric][2].append((epoch, value))

        result_type = self.all_results[metric][0]
        if result_type == 'large':
            if self.best_results[metric] is None or value > self.best_results[metric][0]:
                self.best_results[metric] = (value, epoch)
                self.get_best_results[metric] = True
            else:
                self.get_best_results[metric] = False
        else:
            if self.best_results[metric] is None or value < self.best_results[metric][0]:
                self.best_results[metric] = (value, epoch)
                self.get_best_results[metric] = True
            else:
                self.get_best_results[metric] = False
    

    def print_current_result(self, epoch):
        valid_metrics = [data[0] for data in self.result_info]
        print("Test of epoch: {} complete".format(epoch))
        print_content = ""
        for metric in valid_metrics:
            result_type, scale_ratio, results = self.all_results[metric]
            print_content += f"{metric}:{results[-1][1]*scale_ratio:.3f} "
        print(print_content.strip())


    def print_best_results(self):
        valid_metrics = [data[0] for data in self.result_info]
        # record_content = ""
        for metric in valid_metrics:
            scale_ratio = self.all_results[metric][1]
            best_result, best_epoch = self.best_results[metric]
            best_result *= scale_ratio
            print(f"{metric} : {best_result:.3f} (epoch : {best_epoch})")
            # record_content += f"{best_result:.3f} ({best_epoch}) / "
        # record_content = record_content[:-3] # remove the last " / "
        # print(record_content)
    

    def achieve_better(self):
        return self.get_best_results['inter_mpjpe_3d']