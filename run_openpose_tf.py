"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import cv2
import numpy as np
import argparse
import sys
import time
from enum import Enum
# import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

import tf_pose.pafprocess.pafprocess as pafprocess

from acllite.acllite_resource import AclLiteResource 
from acllite.acllite_model import AclLiteModel

MODEL_PATH = os.path.join("/home/HwHiAiUser/3DPoseEstimation/model/OpenPose_for_TensorFlow_BatchSize_1.om")
IMAGE_PATH = "data/000009.jpg"

print("MODEL_PATH:", MODEL_PATH)

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _round(v):
        return int(round(v))

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:
            # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - self._round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": self._round((x + x2) / 2),
                    "y": self._round((y + y2) / 2),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}
        else:
            return {"x": self._round(x),
                    "y": self._round(y),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8

        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -
        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        return {"x": self._round((x + x2) / 2),
                "y": self._round((y + y2) / 2),
                "w": self._round(x2 - x),
                "h": self._round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    
def nms(heatmaps):
    results = np.empty_like(heatmaps)
    for i in range(heatmaps.shape[-1]):
        heat = heatmaps[:,:,i]
        hmax = pool2d(heat, 3, 1, 1)
        keep = (hmax == heat).astype(float)

        results[:, :, i] = heat * keep
    return results

def estimate_paf(peaks, heat_mat, paf_mat):
    pafprocess.process_paf(peaks, heat_mat, paf_mat)

    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                pafprocess.get_part_score(c_idx)
            )

        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans

def draw(img, humans):
    # high_keypoints = dict()
    # vote for the highest point for each keypoint
    # for human in humans:
    #     for key, body_part in human.body_parts.items():
    #         if body_part.score < 0.5:
    #             continue
            
    #         if key in high_keypoints and high_keypoints[key].score < body_part.score:
    #             high_keypoints[key] = body_part
    #         else:
    #             high_keypoints[key] = body_part

    print()
    # delete score < 0.5 from all humans
    # for key in range(17):
    #     if key in high_keypoints and high_keypoints[key].score < 0.5:
    #         del high_keypoints[key]
    
    # print(high_keypoints.items())

    # keypoints = dict()
    # h, w, _ = img.shape
    # for key, body_part in high_keypoints.items():
    #     if key <= 13:
    #         keypoints[key] = (float(w*body_part.x), float(h*body_part.y))
    # raise Exception

    keypoints = dict()
    h, w, _ = img.shape
    for human in humans:
        for key, body_part in human.body_parts.items():
            # not adding excessive points
            if key <= 13:
                keypoints[key] = (float(w*body_part.x), float(h*body_part.y))

    return keypoints

def post_process(heat):
    heatMat = heat[:,:,:19]
    pafMat = heat[:,:,19:]
    
    # ''' Visualize Heatmap '''
    # print(heatMat.shape, pafMat.shape)
    # for i in range(19):
    #     plt.imshow(heatMat[:,:,i])
    # plt.savefig("outputs/heatMat.png")

    blur = cv2.GaussianBlur(heatMat, (25, 25), 3)

    peaks = nms(heatMat)
    humans = estimate_paf(peaks, heatMat, pafMat)
    return humans

def pre_process(img, height=368, width=656):
    model_input = cv2.resize(img, (width, height))

    return model_input[None].astype(np.float32).copy()

def main(img, model):
    """main"""

    model_input = pre_process(img)

    output = model.execute([model_input])

    humans = post_process(output[0][0])

    # only detect current person.
    if len(humans) >= 2:
        # print()
        # hl = len(humans) 
        # print(hl)
        # for idx in range(hl):
        #     # print("Human length:", len(human))
        #     print()
        #     for key, body_part in humans[idx].body_parts.items():
        #         print(body_part.score, type(body_part))
        #     print()
        
        # take the longest keypoint list
        max_score = 0 # global max
        store = [None]
        for human in humans:
            score = 0 # local max
            for key, body_part in human.body_parts.items():
                score += body_part.score # add up all score of current human
            if max_score <= score:
                max_score = max(max_score, score) # update human score
                store[0] = human # update human object

        humans = store
        

    # generate joints keypoints
    keypoints = draw(img, humans)
    reorder_keypoints = dict()
    # resort keypoints order
    for key, val in keypoints.items():
        if key-2 == -2: reorder_keypoints[12] = val
        elif key-2 == -1: reorder_keypoints[13] = val
        else: reorder_keypoints[key-2] = val

    # reform reordered keypoints list.
    missing = [] # missing value list
    keypoints = []
    for idx in range(0, 14):
        if idx in reorder_keypoints:
            keypoints.append(np.asarray([reorder_keypoints[idx][0], reorder_keypoints[idx][1]]))
        else:
            keypoints.append(np.asarray([0, 0]))
            missing.append(idx) # adding missing keypoints to missing list

    
    return keypoints, missing
