import os
import cv2
import numpy as np

from common.pose_decode import decode_pose
from acllite.acllite_model import AclLiteModel

heatmap_width = 92
heatmap_height = 92

class ModelProcessor:
    def __init__(self, params):
        self.params = params
        self._model_width = params['width']
        self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, img_original):
        
        #preprocess image to get 'model_input'
        model_input = self.preprocess(img_original)

        # execute model inference
        result = self.model.execute([model_input]) 

        # print(result[0].shape)
        # postprocessing: use the heatmaps (the second output of model) to get the joins and limbs for human body
        # Note: the model has multiple outputs, here we used a simplified method, which only uses heatmap for body joints
        #       and the heatmap has shape of [1,14], each value correspond to the position of one of the 14 joints. 
        #       The value is the index in the 92*92 heatmap (flatten to one dimension)
        heatmaps = result[0]
        # calculate the scale of original image over heatmap, Note: image_original.shape[0] is height
        scale = np.array([img_original.shape[1] / heatmap_width, img_original.shape[0]/ heatmap_height])

        canvas, joint_list = decode_pose(heatmaps[0], scale, img_original)

        return canvas, joint_list

    def preprocess(self,img_original):
        '''
        preprocessing: resize image to model required size, and normalize value between [0,1]
        '''
        scaled_img_data = cv2.resize(img_original, (self._model_width, self._model_height))
        preprocessed_img = np.asarray(scaled_img_data, dtype=np.float32) / 255.
        
        return preprocessed_img


