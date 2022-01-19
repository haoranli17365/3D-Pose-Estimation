import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append("./acllite")

from acllite.acllite_resource import AclLiteResource 

from processor_2d_3d import ModelProcessor as Processor_2D_to_3D
from processor_img_2d import ModelProcessor as Processor_Img_to_2D
from common.pose_decode import body_pose_to_h36

MODEL_IMG_2D_PATH = "model/OpenPose_light.om"
MODEL_2D_3D_PATH = "model/pose3d_rie_sim.om"
INPUT_VIDEO = "data/pose3d_test_10s.mp4"

def run_img_to_2d(model_path, input_video_path):
    model_parameters = {
        'model_dir': model_path,
        'width': 368, 
        'height': 368, 
    }
    
    model_processor = Processor_Img_to_2D(model_parameters)
    cap = cv2.VideoCapture(input_video_path)
    keypoints = []
    output_canvases = []

    ret, img_original = cap.read()
    
    img_shape = img_original.shape
    cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    while(ret):
        cnt += 1; print(end=f"\rImg to 2D Prediction: {cnt} / {total_frames}")

        canvas, joint_list = model_processor.predict(img_original)
        keypoints.append(joint_list)
        output_canvases.append(canvas)

        all_frames.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

        ret, img_original = cap.read()
    print()
    keypoints = np.asarray(keypoints)
    keypoints = body_pose_to_h36(keypoints)
    cap.release()
    
    return keypoints, img_shape, all_frames


def run_2d_to_3d(model_path, keypoints, input_video_path, output_video_dir, output_format, img_shape, all_frames):
    
    model_parameters = {
        'model_dir': model_path,
        'cam_h': img_shape[0],
        'cam_w': img_shape[1]
    }

    model_processor = Processor_2D_to_3D(model_parameters)
    
    output = model_processor.predict(keypoints)

    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_path = f'{output_video_dir}/output-{input_filename}.{output_format}'

    model_processor.generate_visualization(keypoints, output, input_video_path, video_output_path, all_frames)
    print("Output exported to {}".format(video_output_path))

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