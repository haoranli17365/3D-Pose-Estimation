import cv2
import numpy as np
import math
import os, sys
sys.path.append('../..')

heatmap_width = 92
heatmap_height = 92

"""
Joints Explained
14 joints:
0-right shoulder, 1-right elbow, 2-right wrist, 3-left shoulder, 4-left elbow, 5-left wrist, 
6-right hip, 7-right knee, 8-right ankle, 9-left hip, 10-left knee, 11-left ankle, 
12-top of the head and 13-neck

                     12                     
                     |
                     |
               0-----13-----3
              /     / \      \
             1     /   \      4
            /     /     \      \
           2     6       9      5
                 |       |
                 7       10
                 |       |
                 8       11
"""

JOINT_LIMB = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]
COLOR = [[0, 255, 255], [0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0], [0, 0, 255], [255, 0, 0],[255, 0, 0],[255, 0, 0], [255, 0, 0]]

def decode_pose(heatmaps, scale, image_original):

    # obtain joint list from heatmap
    # joint_list: a python list of joints, joint_list[i] is an numpy array with the (x,y) coordinates of the i'th joint (refer to the 'Joints Explained' in this file, e.g., 0th joint is right shoulder)  
    joint_list = [peak_index_to_coords(heatmap)*scale for heatmap in heatmaps]
    
    # print(joint_list)
    # plot the pose on original image
    canvas = image_original
    for idx, limb in enumerate(JOINT_LIMB):
        joint_from, joint_to = joint_list[limb[0]], joint_list[limb[1]]
        canvas = cv2.line(canvas, tuple(joint_from.astype(int)), tuple(joint_to.astype(int)), color=COLOR[idx], thickness=4)
    
    return canvas, joint_list   


def peak_index_to_coords(peak_index):
    '''
    @peak_index is the index of max value in flatten heatmap
    This function convert it back to the coordinates of the original heatmap 
    '''
    peak_coords = np.unravel_index(int(peak_index),(heatmap_height, heatmap_width))
    return np.flip(peak_coords)


def body_pose_to_h36(body_pose_keypoints):
    # Body Pose Order
    # 0-right shoulder, 1-right elbow, 2-right wrist, 3-left shoulder, 4-left elbow, 5-left wrist, 
    # 6-right hip, 7-right knee, 8-right ankle, 9-left hip, 10-left knee, 11-left ankle, 
    # 12-top of the head and 13-neck 

    #H36M Order
    # PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    # L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    # L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16

    h36_keypoints = []
    for i in range (0, body_pose_keypoints.shape[0]):
        keypoints = np.zeros([17, 2])
        
        # Pelvis
        keypoints[0][0] = (body_pose_keypoints[i][6][0] + body_pose_keypoints[i][9][0]) / 2
        keypoints[0][1] = (body_pose_keypoints[i][6][1] + body_pose_keypoints[i][9][1]) / 2

        # Right Hip
        keypoints[1] = body_pose_keypoints[i][6]

        # Right Knee
        keypoints[2] = body_pose_keypoints[i][7]

        # Right Foot
        keypoints[3] = body_pose_keypoints[i][8]

        # Left Hip
        keypoints[4] = body_pose_keypoints[i][9]

        # Left Knee
        keypoints[5] = body_pose_keypoints[i][10]

        # Left Foot
        keypoints[6] = body_pose_keypoints[i][11]

        # Spine
        keypoints[7][0] = (body_pose_keypoints[i][6][0] + body_pose_keypoints[i][9][0] + body_pose_keypoints[i][0][0] + body_pose_keypoints[i][3][0]) / 4
        keypoints[7][1] = (body_pose_keypoints[i][6][1] + body_pose_keypoints[i][9][1] + body_pose_keypoints[i][0][1] + body_pose_keypoints[i][3][1]) / 4

        # Thorax
        keypoints[8] = body_pose_keypoints[i][13]

        # Nose
        keypoints[9][0] = (body_pose_keypoints[i][12][0] + body_pose_keypoints[i][13][0]) / 2
        keypoints[9][1] = (body_pose_keypoints[i][12][1] + body_pose_keypoints[i][13][1]) / 2

        # Head
        keypoints[10] = body_pose_keypoints[i][12]

        # Left Shoulder
        keypoints[11] = body_pose_keypoints[i][3]

        # Left Elbow
        keypoints[12] = body_pose_keypoints[i][4]

        # Left Wrist
        keypoints[13] = body_pose_keypoints[i][5]

        # Right Shoulder
        keypoints[14] = body_pose_keypoints[i][0]

         # Right Elbow
        keypoints[15] = body_pose_keypoints[i][1]

         # Right Wrist
        keypoints[16] = body_pose_keypoints[i][2]

        h36_keypoints.append(keypoints)

    return np.asarray(h36_keypoints)
