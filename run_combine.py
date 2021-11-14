import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append("./acllite")
from acllite.acllite_model import AclLiteModel
from acllite.acllite_resource import AclLiteResource 
import run_openpose_tf # import run_openpose_tf
from processor_2d_3d import ModelProcessor as Processor_2D_to_3D
from processor_img_2d import ModelProcessor as Processor_Img_to_2D
from common.pose_decode import body_pose_to_h36
import math
MODEL_IMG_2D_PATH = "model/OpenPose_for_TensorFlow_BatchSize_1.om"
MODEL_2D_3D_PATH = "model/video_pose_3d.om"
INPUT_VIDEO = "data/pose3d_test_10s.mp4"

MISSING_RANGE = 45 # polynomial fitting range.

def run_img_to_2d(model_path, input_video_path):
    # init model
    model_processor = AclLiteModel(model_path)

    # parse frames to images
    cap = cv2.VideoCapture(input_video_path)
    keypoints = []
    # output_canvases = []

    ret, img_original = cap.read()
    try:
        img_shape = img_original.shape
    except:
        raise Exception("Invalid Video Input Given.")
    cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    missing_dict = dict()
    while(ret):
        cnt += 1; print(end=f"\rImg to 2D Prediction: {cnt} / {total_frames}")
        joint_list, missing = run_openpose_tf.main(img_original, model_processor)
        # adding missing points to the list.
        if missing: missing_dict[cnt-1] = missing
        # canvas, joint_list = run_openpose_tf.main(img_original, model_processor)
        keypoints.append(joint_list)

        # all_frames.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

        ret, img_original = cap.read()
    print()

    # find missing keypoints.
    total_missing = len(missing_dict)
    cnt = 0
    for key, val in missing_dict.items():
        cnt += 1; print(end=f"\rFitting missing points with Polynomial Fitting: {cnt} / {total_missing}")
        for pt in val:
            m_x, m_y = [], []
            idx_list_x, idx_list_y = [], []
            for idx in range(1, MISSING_RANGE+1): 
                # check key still in-bound
                if (key - idx) >= 0 and keypoints[key-idx][pt][0] != 0.0:
                    m_x.append(keypoints[key-idx][pt][0])
                    idx_list_x.append(key-idx)

                if (key - idx) >= 0 and keypoints[key-idx][pt][1] != 0.0:
                    m_y.append(keypoints[key-idx][pt][1])
                    idx_list_y.append(key-idx)
                    

                if (key + idx) < total_frames and keypoints[key+idx][pt][0] != 0.0:
                    m_x.append(keypoints[key+idx][pt][0])
                    idx_list_x.append(key+idx)


                if (key + idx) < total_frames and keypoints[key+idx][pt][1] != 0.0:
                    m_y.append(keypoints[key+idx][pt][1])
                    idx_list_y.append(key+idx)
            
            # fill missing points with polynomial fitting
            poly_x = np.polyfit(idx_list_x, m_x, 2)
            poly_y = np.polyfit(idx_list_y, m_y, 2)
            fit_eq_x = poly_x[0] * np.square(key) + poly_x[1] * key + poly_x[2]
            fit_eq_y = poly_y[0] * np.square(key) + poly_y[1] * key + poly_y[2]

            keypoints[key][pt][0] = fit_eq_x
            keypoints[key][pt][1] = fit_eq_y
    print()

    # drawing lines.
    all_frames = []
    cap = cv2.VideoCapture(input_video_path)
    ret, img_original = cap.read()
    idx = 0
    while(ret and idx < len(keypoints)):
        canvas = decode_pose(keypoints[idx], img_original)
        all_frames.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ret, img_original = cap.read()
        idx += 1
    
    keypoints = np.asarray(keypoints)
    keypoints = body_pose_to_h36(keypoints)
    cap.release()

    model_processor.destroy()

    return keypoints, img_shape, all_frames

JOINT_LIMB = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]
COLOR = [[0, 255, 255], [0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0], [0, 0, 255], [255, 0, 0],[255, 0, 0],[255, 0, 0], [255, 0, 0]]

def decode_pose(joint_list, image_original):

    # obtain joint list from heatmap
    # joint_list: a python list of joints, joint_list[i] is an numpy array with the (x,y) coordinates of the i'th joint (refer to the 'Joints Explained' in this file, e.g., 0th joint is right shoulder)  
    # joint_list = [peak_index_to_coords(heatmap)*scale for heatmap in heatmaps]
    # print(joint_list)
    # plot the pose on original image
    canvas = image_original
    for idx, limb in enumerate(JOINT_LIMB):
        joint_from, joint_to = joint_list[limb[0]], joint_list[limb[1]]
        canvas = cv2.line(canvas, tuple(joint_from.astype(int)), tuple(joint_to.astype(int)), color=COLOR[idx], thickness=4)
    
    return canvas


def run_2d_to_3d(model_path, keypoints, input_video_path, output_video_dir, output_format, img_shape, all_frames):
    
    model_parameters = {
        'model_dir': model_path,
        'cam_h': img_shape[0],
        'cam_w': img_shape[1]
    }

    model_processor = Processor_2D_to_3D(model_parameters)
    
    output = model_processor.predict(keypoints)

    detect_pose_signal(output)
    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_path = f'{output_video_dir}/output-{input_filename}.{output_format}'

    model_processor.generate_visualization(keypoints, output, input_video_path, video_output_path, all_frames)
    

    print("Output exported to {}".format(video_output_path))

# detect pose signal.
def detect_pose_signal(output):
    # H36M Order
    # PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    # L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    # L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16
    # print(output)
    l_hip = 0.0
    r_hip = 0.0
    # compare z-index depth for the left hip and right hip
    for idx in range(len(output)):
        l_hip += output[idx][1][2]
        r_hip += output[idx][4][2]

    l_hip_avg = np.mean(l_hip)
    r_hip_avg = np.mean(r_hip)
    print(l_hip_avg, r_hip_avg)
    depth_diff = abs(l_hip_avg - r_hip_avg)
    # left: 0.114, 12.580614662221631
    # stop: 2.391942911250253
    # right: 8.282248636889676
    # print()
    if depth_diff >= 10:
        print('######################################')
        print("left-turn traffic signal is performed.")
        print('######################################')
    elif depth_diff >= 8:
        # arm raising causing the y to decrease as normal.
        print('######################################')
        print("Right-turn traffic signal is performed.")
        print('######################################')
    else:
        print('######################################')
        print("Stop traffic signal is performed.")
        print('######################################')

if __name__ == "__main__":
    
    description = '3D Pose Lifting'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--model2D', type=str, default=MODEL_IMG_2D_PATH)
    parser.add_argument('--model3D', type=str, default=MODEL_2D_3D_PATH)
    parser.add_argument('--input', type=str, default=INPUT_VIDEO)
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output Path")
    parser.add_argument('--output_format', type=str, default='gif', help="Either gif or mp4")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    acl_resource = AclLiteResource()
    acl_resource.init()

    keypoints, img_shape, all_frames = run_img_to_2d(args.model2D, args.input)

    run_2d_to_3d(args.model3D, keypoints, args.input, args.output_dir, args.output_format, img_shape, all_frames)


