import os
import numpy as np

from acllite.acllite_model import AclLiteModel

from common.generators import Evaluate_Generator, UnchunkedGenerator
from common.quaternion import qrot
from common.visualization import render_animation
from common.skeleton import Skeleton

class ModelProcessor:
    
    def __init__(self, params):
        self.params = params
        self._cam_width = params['cam_w']
        self._cam_height = params['cam_h']
        self._kps_left = [4, 5, 6, 11, 12, 13]
        self._kps_right = [1, 2, 3, 14, 15, 16]
        self._h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15],
                                        joints_left=[4, 5, 6, 11, 12, 13],
                                        joints_right=[1, 2, 3, 14, 15, 16])
        self._metadata = {'layout_name': 'h36m',
                          'num_joints': 17,
                          'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]}

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, input_keypoints):
        
        #preprocess image to get 'model_input'
        model_input = self.normalize_screen_coordinates(input_keypoints, self._cam_width, self._cam_height)

        prediction = []
        
        # create a slidiing window to create a prediction
        prediction = np.zeros(shape=(model_input.shape[0], 17, 3))
        for fr in range(model_input.shape[0] - 10 + 1):
            cur_kp = np.array(model_input[fr:fr+10, :, :])
            gen = UnchunkedGenerator(None, None, [cur_kp],
                                pad=121, causal_shift=0, augment=True,
                                kps_left=self._kps_left, kps_right=self._kps_right, 
                                joints_left=self._kps_left, joints_right=self._kps_right)
            _, _, batch_2d = gen.next_epoch()
            
            inputs_2d = batch_2d.astype('float32').copy()
            cur_prediction = np.array(self.model.execute([inputs_2d])[0])

            cur_prediction[1, :, :, 0] *= -1
            cur_prediction[1, :, self._kps_left + self._kps_right] = cur_prediction[1, :, self._kps_right + self._kps_left]
            cur_prediction = np.mean(cur_prediction, axis=0)


            prediction[fr+9, :, :] = cur_prediction[-1, :, :]
            prediction[fr:fr+10, :, :] = ((prediction[fr:fr+10, :, :] * fr) + cur_prediction) / (fr + 1)
        
        return prediction

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2

        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / w * 2 - [1, h/w]
    
    def image_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        
        # Reverse camera frame normalization
        return (X + [1, h/w])*w/2
    
    def camera_to_world(self, X, R, t):
        return qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t

    def generate_visualization(self, input_keypoints, prediction, input_video_path, output_video_path, all_frames):
        rotation_matrix = [ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804]
        t = [ 2.0831823, -4.912173, 1.5610787]
        
        prediction = self.camera_to_world(prediction, R=rotation_matrix, t=t)
        
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        input_keypoints = self.image_coordinates(input_keypoints[..., :2], w=self._cam_width, h=self._cam_height)
        
        render_animation(input_keypoints, {'num_joints': 17, 'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]}, anim_output,
                         self._h36m_skeleton, 30, 3000, 70, output_video_path, (self._cam_width, self._cam_height),
                         limit=-1, downsample=1, size=6,
                         input_video_path=input_video_path,
                         input_video_skip=0, all_frames=all_frames)


