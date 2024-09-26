import torch


# I/O ##################################################################################################################
# Video's path


video_src = "/home/jovyan/Desktop/Pytorch/videos/input/video_2.mp4"  
video_output = "/home/jovyan/Desktop/Pytorch/videos/output/video_2_out.mp4" 

text_output = "/home/jovyan/Desktop/Pytorch/videos/output/video_2_out.csv"  
# DETECTOR #############################################################################################################
compound_coef = 3
force_input_size = None  # set None to use default size

threshold = 0.8
iou_threshold = 0.2

use_cuda = torch.cuda.is_available()
# use_cuda = True
use_float16 = False
cudnn_fastest = True
cudnn_benchmark = True
# coco_name
'''
obj_list = ['person','chair','couch','potted plant','bed', 'dining table','toilet',
                     'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
                     'sink','refrigerator','book',
                     'clock','vase','scissors','teddy bear','hair drier','toothbrush']
                     

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
'''

obj_list = ['person',]

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
# input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# TRACKER ##############################################################################################################
REID_CKPT = "/home/jovyan/Desktop/Pytorch/pretrained_models/ckpt.t7"   #rgb model

#REID_CKPT = "/home/jovyan/Desktop/models_residual_trained/model_epoch_8_acc_80.02_loss_0.721.pth"    # mv residual model
#REID_CKPT = "/home/jovyan/Desktop/models_residual_trained/model_epoch_44_acc_83.28_loss_0.651.pth"    # residual model

#REID_CKPT = "/home/jovyan/Desktop/models_residual_trained/model_epoch_34_acc_68.80_loss_1.143.pth"

MAX_DIST = 0
MIN_CONFIDENCE = 0
NMS_MAX_OVERLAP = 0
MAX_IOU_DISTANCE = 0
MAX_AGE = 50  # as low as possible
N_INIT = 2
NN_BUDGET = 100


# TARGETS
selected_target = [obj_list.index('person'),
                  # obj_list.index('chair'),
                  # obj_list.index('bed')
                   ]
