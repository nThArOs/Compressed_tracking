from torch.backends import cudnn

from PIL import Image

import gc
import os
import numpy as np
import time
import torch
import coviar

from torch.autograd import Variable
import torch.nn.functional as F

#from memory_profiler import profile
#import torch.cuda.profiler as profiler

#from lib.model.nms.nms_wrapper import nms
#from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_one_class, bbox_iou
#from lib.model.utils.blob_single import prep_im_for_blob, prep_mv_for_blob, prep_residual_for_blob
#from lib.utils.misc import resize_im
#from lib.utils.image_viewer import ImageViewer
#from lib.utils.visualization import create_unique_color_uchar
#from lib.tracking.track import Track
#from lib.tracking.detection import Detection
#from lib.tracking import linear_assignment
#from lib.tracking import distance_metric_func
#from lib.tracking.utils import crop_data_for_boxes
#from lib.model.roi_align.roi_align.roi_align import RoIAlign
import matplotlib.pyplot as plt


import cv2
#from eff.backbone import EfficientDetBackbone
#from eff.efficientdet.utils import BBoxTransform, ClipBoxes
#from eff.utils.utils import preprocess, invert_affine, postprocess, preprocess_video_frame,preprocess_video, preprocess_video_frame_residual
import pickle

import sys
import io

#from eff.deep_sort import build_tracker
from eff.config import *




#def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#    tensor = torch._utils._rebuild_tensor(storage, xystorage_offset, size, stride)
 #   tensor.requires_grad = requires_grad
  #  tensor._backward_hooks = backward_hooks
   # return tensor

#torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class Tracker:
    def __init__(self, base_net_model,resd_model, tracking_model, appearance_model, classes=None, args=None, cfg=None, compound_coef=2, compound_coef_resd=3):

        self.classes = np.asarray(['__background__', 'person']) if classes is None else classes
        self.args = args
        self.cfg = cfg
        self.gop_size = 12            ### GOP SIZE
        self.im_viewer = None

        self.feature_crop_size = self.args.feature_crop_size  # (1024, 7, 3) the cropped size of feature map, (h, w)
       # self.mv_crop_size = self.args.mv_crop_size  # mv crop size. (c, h, w)
       # self.im_crop_size = self.args.im_crop_size  # im crop size, (c, h, w)
       # self.resdual_crop_size = self.args.residual_crop_size # residual crop size, (c, h, w)

        # the iou less than (1-self.max_iou_distance) will be disregarded.
        self.max_iou_distance = 0.7
        # the probability of been different targets larger than this value will be disregarded.
        self.max_appearance_distance = 0.25
        # the euclidean distance between the centers of two boxes (normalized by the diagonal line of this frame)
        # that larger than this threshold will be disregarded.
        self.max_euclidean_distance = 0.2 #0.2 utilisee pour l'association par apparence

        self.mot_dir = self.args.mot_dir       # '/workspace/data/MOT'
        self.video_file = None                 # the file path of the video need to track
        self.dataset_year = None  # MOT16, MOT17
        self.phase = None  # 'train', 'test'
        self.seq = None  # the name of sequence  : MOT-02-FRCNN .....
        self.frame_id = None  # the frame id of current frame
        self.detector_name = 'EFFICIENTDET'
        self.tracking_thr = {'EFFICIENTDET': 0.5, 'DEEPSORT': 0.97}

        self._next_id = 0  # the track id
        self.tracks = []  # list, used to save tracks (track.Track)
        # list, used to save the detections (detection.Detection) need to track in current frame
        self.detections_to_track = []
        self.detections_to_save = []
        self.tracking_results = None  # used to save the tracking results
        self.detection_results = None  # used to save the detection results
        self.public_detections = None  # used to save the loaded pre_detections : OFFLINE

        # define some variables used to do time analysis
        self.tracked_seqs = []
        self.num_frames = []  # used to store the number of frames for each video
        self.load_time = []  # used to store the time consumption of loading the image from the video
        self.detect_time = []  # used to store the time consumption of detecting
        self.associate_time = []  # used to store the time consumption of association targets with detection
        self.track_time = []  # used to store the time consumption of tracking regression
        self.offset_time = []  # used to store the time consumption of doing offsets
        self.pre_boxes_list = [] # used to store the tracked boxes in previous frames
        self.pre_boxes_list_history = 12 # the number of previous frames

        # define the mean and std for bounding-box regression
        self.bbox_reg_std = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.bbox_reg_mean = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        if self.args.cuda:
            self.bbox_reg_std = self.bbox_reg_std.cuda()
            self.bbox_reg_mean = self.bbox_reg_mean.cuda()

        # define the network
        self.base_net_model = base_net_model                # DETECTION MODEL
        self.tracking_model = tracking_model
        self.appearance_model = appearance_model

        self.resd_model = resd_model   # residual detection model
        self.compound_coef_resd = compound_coef_resd
        # define some tools for RoIAlign cropping        # réutilisable
#        self.im_roi_align = RoIAlign(crop_width=self.im_crop_size[2], crop_height=self.im_crop_size[1],
 #                                    transform_fpcoor=True)
        self.feat_roi_align = RoIAlign(crop_width=self.feature_crop_size[2], crop_height=self.feature_crop_size[1],
                                       transform_fpcoor=False)

        self.roi_align_box_index = torch.zeros(500).int()
        if self.args.cuda:
            self.roi_align_box_index = self.roi_align_box_index.cuda()
        self.roi_align_box_index = Variable(self.roi_align_box_index)

        self.cost_matrix = torch.zeros((100, 100)).float().fill_(linear_assignment.INFTY_COST).cuda()

        # define some variables used to store the input data
        self.im_data = torch.FloatTensor(1)  # used for detection or tracking
        self.im_info = torch.FloatTensor(1)  # used for detection, [h, w, im_scale, frame_type]
        self.boxes = torch.FloatTensor(1)  # used for detection or tracking
        self.num_boxes = torch.FloatTensor(1) # used for detection
        self.track_feature = torch.FloatTensor(1) # used for appearance matching    ( frame t-1)
        self.detection_feature = torch.FloatTensor(1) # used for appearance matching  ( frame courante : t)

        if self.args.cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.boxes = self.boxes.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.track_feature = self.track_feature.cuda()
            self.detection_feature = self.detection_feature.cuda()

        self.im_data = Variable(self.im_data, volatile=True)
        # the info of this frame [1, 4], [im_h, im_w, im_scale, frame_type]
        self.im_info = Variable(self.im_info, volatile=True)
        self.boxes = Variable(self.boxes, volatile=True)
        self.num_boxes = Variable(self.num_boxes, volatile=True)
        self.track_feature = Variable(self.track_feature, volatile=True)
        self.detection_feature = Variable(self.detection_feature, volatile=True)

        self.image_size = Variable(torch.FloatTensor([0, 0]))


        self.threshold = 0.8       # min_conf pour eff nonkey
        self.iou_threshold = 0.3  #nms pour efficientdet non key

        self.use_cuda = use_cuda
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True

        self.obj_list = obj_list

        # TRACKER
        self.reid_cpkt = REID_CKPT
        self.max_dist = 0.2
        self.min_confidence = 0.97 ###################################################################
        self.nms_max_overlap = 0.8
        self.max_iou_distance_DS = 0.7
        self.max_age = 50
        self.n_init = 2
        self.nn_budget = 100

        # load tracker model,
        self.trackers = []
        self.selected_target = selected_target
        for num in range(0, len(self.selected_target)):
            self.trackers.append(build_tracker(REID_CKPT,
                                               self.max_dist,
                                               self.min_confidence,
                                               self.nms_max_overlap,
                                               self.max_iou_distance_DS,
                                               self.max_age,
                                               self.n_init,
                                               self.nn_budget,
                                               use_cuda = True))
        # video frames
        self.frame_id = 0
        self.keyframe = None

        self.compound_coef=compound_coef
        self.key_features=None
        self.feature_scale = None

    def xyxy_to_tlwh(self, bbox):
        # bbox: [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = bbox
        t = x1
        l = y1
        w = x2 - x1
        h = y2 - y1
        return [t, l, w, h]

    def xyxy_to_xywh(self,boxes_xyxy: np.array):
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()
        else:
            raise TypeError

        if boxes_xyxy.ndim == 1:
            boxes_xyxy = np.expand_dims(boxes_xyxy, axis=0)

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh

    def tlwh_to_xywh(self, boxes_tlwh: np.array):
        if isinstance(boxes_tlwh, torch.Tensor):
            boxes_xywh = boxes_tlwh.clone()
        elif isinstance(boxes_tlwh, np.ndarray):
            boxes_xywh = boxes_tlwh.copy()
        else:
            raise TypeError

        if boxes_tlwh.ndim == 1:
            boxes_tlwh = np.expand_dims(boxes_tlwh, axis=0)

        boxes_xywh[:, 0] = boxes_tlwh[:, 0] + boxes_tlwh[:, 2] / 2.0  # center_x = left + width/2
        boxes_xywh[:, 1] = boxes_tlwh[:, 1] + boxes_tlwh[:, 3] / 2.0  # center_y = top + height/2

        return boxes_xywh


    def prepare_data_to_show(self, in_data, tool_type='cv2'):
        # indata: [bs, c, h, w]
        if isinstance(in_data, Variable):
            in_data = in_data.data
        if in_data.is_cuda:
            in_data = in_data.cpu()

        in_data = in_data[0]
        in_data = in_data.permute(1, 2, 0) # h w c, BGR channel

        if tool_type == 'cv2':
            pass
        elif tool_type == 'plt':
            in_data = in_data[:, :, [2, 1, 0]]

        in_data = np.asanyarray(in_data.numpy(), dtype=np.uint8)
        return in_data

    def _crop_data_for_boxes(self, boxes, in_data, scale=None, in_data_type='feature'):
        if not isinstance(in_data, Variable):
            in_data = Variable(in_data)

        if not isinstance(boxes, Variable):
            boxes = Variable(boxes)   # x1,y1 x2,y2

        if len(boxes.size()) == 1:
            boxes = boxes.unsqueeze(dim=0)

        if scale is None:
            scale = 1
        else:
            if not isinstance(scale, Variable):
                scale = Variable(scale)
            if len(scale.size()) == 1: # if f_scale has size [4]
                scale = scale.unsqueeze(dim=0) # change to [1, 4]

        box_index = self.roi_align_box_index[:boxes.size(0)]
#        print('boxes',boxes)
   #     print('scale TEST',scale)
        boxes = boxes.type(torch.cuda.FloatTensor)
        boxes = boxes * scale
 #       print('boxes scaled',boxes)
        if in_data_type == 'feature':
            croped_data = self.feat_roi_align(in_data, boxes, box_index)

        print('croped_data', croped_data.shape)  # devrait imprimer torch.Size([1, 560, 9, 9])
     #   print('croped_data TEST',croped_data)
        return croped_data




    def reset(self, tracking_output_file=None, detection_output_file=None):    # save les résultats de détection et tracking et faire un reset pour la prochaine vidéo
        """
        This function reset the tracker, so it can track on the next video
        :param tracking_output_file: the file path to save tracking results
        :param detection_output_file: the file path to save detection results
        :return:
        """

        # before reset the tracker, we first save the tracking and detection results
        fmt = ['%d', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%d', '%d', '%d']  # the format to save the results
       # print('self.tracking_results',self.tracking_results)

        if self.args.cuda:
            if self.tracking_results is None:
                self.tracking_results = torch.zeros(1, 10)
            else:
                self.tracking_results = self.tracking_results.cpu()

            if self.detection_results is None:
                self.detection_results = torch.zeros(1, 10)
            else:
                self.detection_results = self.detection_results.cpu()

        # save the detection as pth file
        if tracking_output_file is not None:
            if os.path.exists(tracking_output_file):
                os.remove(tracking_output_file)
            tracking_results = self.tracking_results.numpy()
            np.savetxt(tracking_output_file, tracking_results, fmt=fmt, delimiter=',')
        #    print('tracking_results',tracking_results)
            print('tracking results saved in ' + tracking_output_file)

        if detection_output_file is not None and self.detection_results is not None:
            if self.args.save_detections_with_feature:
                detection_output_file_pth = detection_output_file.split('txt')
                detection_output_file_pth = detection_output_file_pth[0] + 'pth'
                torch.save(self.detection_results.contiguous(), detection_output_file_pth)
                print('detections results saved in ' + detection_output_file)

            if os.path.exists(detection_output_file):
                os.remove(detection_output_file)
            detection_results = self.detection_results.numpy()
            np.savetxt(detection_output_file, detection_results[:, 0:10], fmt=fmt, delimiter=',')
            print('detections results saved in ' + detection_output_file)

        # reset the tracker
        self.im_viewer = None
        self.frame_id = None
        self._next_id = 0  # the track id
        self.tracks = []  # list, used to save tracks (track.Track)
        self.tracking_results = None  # used to save the tracking results
        self.detections_to_track = []  # list, used to save the detections (detection.Detection) of current frame
        self.detections_to_save = []
        self.detection_results = None  # used to save the detection results
        self.pre_boxes_list = []


    def save_time(self, time_file):
        """
        This function save the time collected untill this function is called.
        :param time_file:
        :return:
        """
        # time analysis
        if len(self.tracked_seqs) == 0:
            seqs = np.array([''])
        else:
            seqs = np.array(self.tracked_seqs)

        self.tracked_seqs = []

        if len(self.num_frames) == 0:
            num_frames = np.array([0])
        else:
            num_frames = np.array(self.num_frames)

        self.num_frames = []

        if len(self.load_time) == 0:
            load_time = np.array([0])
        else:
            load_time = np.array(self.load_time)

        self.load_time = []

        if len(self.detect_time) == 0:
            detect_time = np.array([0])
        else:
            detect_time = np.array(self.detect_time)

        self.detect_time = []

        if len(self.associate_time) == 0:
            associate_time = np.array([0])
        else:
            associate_time = np.array(self.associate_time)

        self.associate_time = []

        if len(self.track_time) == 0:
            track_time = np.array([0])
        else:
            track_time = np.array(self.track_time)

        self.track_time = []

        if len(self.offset_time) == 0:
            offset_time = np.array([0])
        else:
            offset_time = np.array(self.offset_time)

        self.offset_time = []

        total_time_load = load_time.sum() + detect_time.sum() + associate_time.sum() + track_time.sum() + offset_time.sum()
        total_frames = len(load_time)

        total_time_no_load = detect_time.sum() + associate_time.sum() + track_time.sum() + offset_time.sum()

        if os.path.exists(time_file):
            os.remove(time_file)
        f = open(time_file, 'w')

        print('sequences:\n', seqs,
              '\n\nnumber of frames:\n', num_frames,
              '\n\ntotal number frames: {}'.format(num_frames.sum()),
              '\n\naverage load time: {}/{} = {}s'.format(load_time.sum(), load_time.shape[0], load_time.mean()),
              '\n\naverage detect time: {}/{} = {}s'.format(detect_time.sum(), detect_time.shape[0],
                                                            detect_time.mean()),
              '\n\naverage associate time: {}/{} = {}s'.format(associate_time.sum(), associate_time.shape[0],
                                                               associate_time.mean()),
              '\n\naverage track time: {}/{} = {}'.format(track_time.sum(), track_time.shape[0], track_time.mean()),
              '\n\naverage offset time: {}/{} = {}'.format(offset_time.sum(), offset_time.shape[0], offset_time.mean()),
              '\n\naverage time per frame (with load) = {}/{} = {}'.format(total_time_load, total_frames,
                                                                           total_time_load / total_frames),
              '\naverage time per frame (without load) = {}/{} = {}'.format(total_time_no_load, total_frames,
                                                                            total_time_no_load / total_frames),
              '\n\nFPS (with load): {}'.format(1.0 / (total_time_load / total_frames)),
              '\nFPS (without load): {}'.format(1.0 / (total_time_no_load / total_frames)),
              file=f)

        print('time analysis file saved in ' + time_file)
        print('FPS (without load): {}'.format(1.0 / (total_time_no_load / total_frames)))

#    @profile
    def get_frame_blob_from_video(self, video_path, frame_id, load_data_for=None):
        """
        This function extract the image, motion vector, residual from the given video
        and the frame id.
        :param video_path: string, the path to the raw video (.mp4)
        :param frame_id: int, the frame id
        :param load_data_for: str, determine to load the data for tracking or detection
        :return: blob: 3D array, [h, w, 3+2+3]
        :return: im_scale: float(target_size) / float(im_size_min)
        """


        accumulate = False

        gop_idx = int((frame_id - 1) / self.gop_size)  # GOP starts from 0, while frame_id  here starts from 1.
        in_group_idx = int((frame_id - 1) % self.gop_size)  # the index in the group

        if load_data_for == 'track': # load mv and residual
            mv = coviar.load(video_path, gop_idx, in_group_idx, 1, accumulate)
            residual = coviar.load(video_path, gop_idx, in_group_idx, 2, accumulate)
  #          residual = (residual*255 - np.mean(residual*255)) / np.std(residual*255)
   #         residual = np.clip(residual, 0, 1)  # assurez-vous que les valeurs sont entre 0 et 1
    #        residual_shape = residual.shape
   #         frame_data = np.zeros((residual_shape[0], residual_shape[1], 2 + 3))
  #          frame_data[:, :, 0:2] = mv
 #           frame_data[:, :, 2:5] = residual
#            return frame_data, 1
            # check whether it is a gray image
            if len(residual.shape) == 2:
                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            residual_shape = residual.shape
            residual_size_min = np.min(residual_shape[0:2])
            residual_size_max = np.max(residual_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            residual_scale = float(target_size) / float(residual_size_min)
            if np.round(residual_scale * residual_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(residual_size_max)
                target_size = np.round(im_scale * residual_size_min)

            mv, mv_scale = prep_mv_for_blob(im=mv,
                                            mv_normal_scale=self.cfg.MV_NORMAL_SCALE,
                                            mv_means=self.cfg.MV_MEANS,
                                            mv_stds=self.cfg.MV_STDS,
                                            target_size=target_size,
                                            channel=self.cfg.MV_CHANNEL)
            residual, residual_scale = prep_residual_for_blob(im=residual,
                                                              pixel_normal_scale=self.cfg.RESIDUAL_NORMAL_SCALE,
                                                              pixel_means=self.cfg.RESIDUAL_MEANS,
                                                              pixel_stds=self.cfg.RESIDUAL_STDS,
                                                              target_size=target_size,
                                                              channel=self.cfg.RESIDUAL_CHANNEL)

            # check the scales of im, mv and residual
            if mv_scale != residual_scale:
                raise RuntimeError(
                    'the scales to resize motion vector {} and residual {} are not the same'.
                        format(mv_scale, residual_scale))

            residual_shape = residual.shape
            if self.args.tracking_net_data_type == 'mv_residual':
                frame_data = np.zeros((residual_shape[0], residual_shape[1], 2 + 3))
                frame_data[:, :, 0:2] = mv
                frame_data[:, :, 2:5] = residual
            elif self.args.tracking_net_data_type == 'mv':
                frame_data = mv
            elif self.args.tracking_net_data_type == 'residual':
                frame_data = residual

      #      residual_pil = Image.fromarray((residual * 255).astype(np.uint8))  # convertir en PIL image pour sauvegarder
 #           residual_pil.save(f'resd/residual_yarabi{self.frame_id}.jpg')

#            directory = f'ds_dataset/mv_files/{self.seq}'
            directory1 = f'ds_dataset/residual_images/{self.seq}'
  #          if not os.path.exists(directory):
   #             os.makedirs(directory)

            if not os.path.exists(directory1):
                os.makedirs(directory1)

            filename = str(self.frame_id).zfill(6)

     #       with open(f'ds_dataset/mv_files/{self.seq}/{filename}.pkl', 'wb') as f:
      #          pickle.dump(mv, f)
#            np.save(f'ds_dataset/mv_files/{self.seq}/{filename}.npy', mv)
            residual_pil = Image.fromarray((residual * 255).astype(np.uint8))
            residual_pil.save(f'ds_dataset/residual_images/{self.seq}/{filename}.png')



            return frame_data, residual_scale

        elif load_data_for in ['base_feat', 'detect']: # load im (processed)
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)
            return im, 1
            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
           # im_scale = float(target_size) / float(im_size_min)
            im_scale=1.6875
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)
                target_size = np.round(im_scale * im_size_min)

       #    target_size=768
            print('im_shape',im_shape)
            print('im_size_min',im_size_min)
            print('im_size_max',im_size_max)
            print('target_size',target_size)
            print('im_scale',im_scale)

            im_data, im_scale = prep_im_for_blob(im=im,
                                                 pixel_normal_scale=self.cfg.PIXEL_NORMAL_SCALE,
                                                 pixel_means=self.cfg.PIXEL_MEANS,
                                                 pixel_stds=self.cfg.PIXEL_STDS,
                                                 target_size=target_size,
                                                 channel=self.cfg.PIXEL_CHANNEL)
            return im, im_scale

        elif load_data_for == 'vis': # load im (no processed)
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)
            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

            im = resize_im(im, im_scale)
            return im
        elif load_data_for == 'crop': # load im (no processed) and mv
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)
            mv = coviar.load(video_path, gop_idx, in_group_idx, 1, accumulate)
            residual = coviar.load(video_path, gop_idx, in_group_idx, 2, accumulate)
            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

            im = resize_im(im, im_scale)
            residual = resize_im(residual, im_scale)
            mv = mv / im_scale
            mv = resize_im(mv, im_scale)

            return im, mv, residual

    def visualize_results(self, kind_of_boxes='tracking'):
        """
        This function show tracking or detection results in real time.
        :param kind_of_boxes: str, 'tracking', 'detection' or 'both'
        :return:
        """

        # get boxes
        boxes = []
        if kind_of_boxes in ['tracking', 'tracks']:
            for t in self.tracks:
                if t.is_confirmed():
                    tlwh = t.to_tlwh()
                    one_box = [t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(t.confidence, 3)]
                    boxes.append(one_box)
        elif kind_of_boxes in ['detection', 'detections']:
            for d in self.detections_to_track:
                    tlwh = d.to_tlwh()
                    one_box = [-1, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(d.confidence, 3)]
                    boxes.append(one_box)
        else:
            for t in self.tracks:
                if t.is_confirmed():
                    tlwh = t.to_tlwh()
                    one_box = [t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(t.confidence, 3)]
                    boxes.append(one_box)

            for d in self.detections_to_track:
                tlwh = d.to_tlwh()
                one_box = [-1, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(d.confidence, 3)]
                boxes.append(one_box)

        # prepare image
        im_data = self.get_frame_blob_from_video(video_path=self.video_file,
                                                 frame_id=self.frame_id,
                                                 load_data_for='vis')
        im_data = np.asarray(im_data, dtype=np.uint8)

        if self.im_viewer is None:
            im_shape = im_data.shape
            h, w = im_shape[0], im_shape[1]
            self.im_viewer = ImageViewer(update_ms=1, window_shape=(w, h))

        self.im_viewer.image = im_data.copy()
        self.im_viewer.annotate(20, 30, str(self.frame_id) + '/' + str(self.num_frames[-1]), color=(159, 255, 84))

        boxes = np.asarray(boxes)
        self.im_viewer.thickness = 2
        for box in boxes:
            target_id = int(box[0])
            tlwh = box[1: 5]
            if len(box) > 5:
                confidence = str(box[5])
            else:
                confidence = None

            if target_id <= 0:  # detection
                # self.viewer.color = create_unique_color_uchar(random.randint(-100, 100))
                self.im_viewer.color = 0, 0, 255
                self.im_viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_br=confidence)
            else:  # gt or track results
                self.im_viewer.color = create_unique_color_uchar(target_id)
                self.im_viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_tl=str(target_id),
                                         label_br=confidence)

        # show image
        # self.im_viewer.run()
        self.im_viewer.show_image()

    def _get_previous_boxes(self):
        """
        This function get the boxes in previous frames
        :return: 3D variable or None
        """
        # get the boxes in previous frame
        pre_boxes = None
        for t in self.tracks:
            one_box = t.to_tlbr()
            # rescale to origin image
            # one_box = one_box / im_scale
            one_box = one_box.unsqueeze(dim=0)  # [1, 4], [x1, y1, x2, y2]
            batch_indx = one_box.new([[0]])
            one_box = torch.cat((batch_indx, one_box), dim=1)

            if pre_boxes is None:
                pre_boxes = one_box.new().resize_(0, 4)

            pre_boxes = torch.cat((pre_boxes, one_box), dim=0)  # [num_tracks, 4]
        if pre_boxes is not None:
            pre_boxes = pre_boxes.unsqueeze(dim=0)  # [bs ,num_track, 4], here bs = 1
            self.pre_boxes_list.append(pre_boxes)

        if len(self.pre_boxes_list) > 0:
            self.pre_boxes_list = self.pre_boxes_list[-self.pre_boxes_list_history:]
            pre_boxes = torch.cat(self.pre_boxes_list, dim=1)
            self.boxes.data.resize_(pre_boxes.size()).copy_(pre_boxes).contiguous()
            pre_boxes = self.boxes.clone() # shift to GPU if necessary and change to Variable

        return pre_boxes

##################################################################################################################################################################

    def eff_det_key(self):  # self : l'instance de l'objet tracker

        compound_coef = self.compound_coef # ce parametre represente la version de modele efficientdet
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        use_cuda = True
        use_float16 = False
        gpu = 0
        threshold = 0.5
        nms_threshold = 0.2

        params = {       # pour le modele de detection
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
            'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]',
        }



        # Initialiser les objets pour le post-processing
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()


        t1 = time.time()


        im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file, frame_id=self.frame_id, load_data_for='detect')

#        print('im_data_tmp',im_data_tmp.shape)
       # print('im_scale_tmp',im_scale_tmp)
        ori_img, framed_img, framed_meta = preprocess_video_frame(im_data_tmp, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])

        load_time = time.time() - t1

        # Récupérer la taille de l'image originale et la taille de l'image redimensionnée
        original_height, original_width = im_data_tmp.shape[:2]
        print(original_height, original_width)
        resized_height, resized_width = framed_img.shape[:2]
        print(resized_height, resized_width)

        self.image_size = Variable(torch.FloatTensor([resized_width , resized_height ]))
        print('self.image_size',self.image_size)
        x = torch.from_numpy(framed_img)

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        t2 = time.time()
        # Passer l'image à travers EfficientDet
        print('x.shape',x.shape)
   #     print('CECI X INPUT', x)

         # FORWRD INFERENCE
        with torch.no_grad():
            features, regression, classification, anchors = self.base_net_model(x)

        # Post-traiter les prédictions
        preds = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold)

        if not preds: # Aucune détection
            pass

        # Inverser les transformations pour récupérer les coordonnées d'origine
        preds = invert_affine([framed_meta], preds)[0]
      #  print('preds.shape',preds.shape)
        # Recuperer les scores, les classes et les boîtes
        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        mask = class_ids == 0
        rois = rois[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]



        det_bboxes = torch.zeros((rois.shape[0], 5))

        if rois.shape[0] > 0:
            det_bboxes[:, :4] = torch.from_numpy(rois)
            det_bboxes[:, 4] = torch.from_numpy(scores)
       # On determine la taille TARGET comme la taille de la plus grande feature map
        target_size = features[2].shape[-2:]  # Assuming features[3] is the largest one

       # On fait un interpolate (upsampling) pour chaque feature map pour atteindre la taille cible, puis on les concatène
        features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
        features = torch.cat(features, dim=1)
        self.key_features=features

        print('combined features shape)',features.shape)

        if det_bboxes.nelement() != 0:
             det_bboxes = det_bboxes.unsqueeze(0)
        print('det_bboxes KEY FRAME',det_bboxes.shape)


        self.image_scale = torch.tensor([
            float(round(resized_width)) / float(original_width),   #768/1920 (eff2)
            float(round(resized_height)) / float(original_height),
            float(round(resized_width)) / float(original_width),
            float(round(resized_height)) / float(original_height),
        ], dtype=torch.float32)

        print('Shape of self.image_scale: ', self.image_scale.shape)

        print('self.image_scale KEY',self.image_scale)
#        print('det_bboxes[:,:,0:4]',det_bboxes[:,:,0:4])
        print('det_bboxes',det_bboxes)
        det_bboxes[:,:,0:4]=det_bboxes[:,:,0:4] * self.image_scale       # revenir a 768 768

        self.feature_scale = torch.tensor([
            float(round(target_size[0])) / float(resized_width),   #24/768 (eff2)
            float(round(target_size[1])) / float(resized_height),
            float(round(target_size[0])) / float(resized_width),
            float(round(target_size[1])) / float(resized_height),
        ], dtype=torch.float32).cuda()
        print('self.feature_scale',self.feature_scale)

        # Convertir les boîtes de détection en listes de détections à sauvegarder et à suivre
        self.detections_to_save, self.detections_to_track = self._detection_bbox_to_detection_list(det_bboxes=det_bboxes,
                                                                                                       feature_map=features,
                                                                                                       f_scale=self.feature_scale) 
        detect_time = time.time() - t2    #le temps de détection

        # print("detecrtion time: {}".format(detect_time))
        return load_time, detect_time


    def eff_det_nonkey(self):  # self : l'instance de l'objet tracker

        use_cuda = True
        use_float16 = False
        gpu = 0


        compound_coef = self.compound_coef_resd  # ce parametre represente la version de modele efficientdet
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


        params = {
            'mean': [0,0,0],
            'std': [1,1,1],
            'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
            'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]',
        }

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()


        t1 = time.time()

        im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file, frame_id=self.frame_id, load_data_for='track')
        print('im_scale_tmp DEBUG',im_scale_tmp)
        print('im_data_tmp DEBUG',im_data_tmp.shape)

        # frame preprocessing
        residual = (im_data_tmp[:, :, 2:5]* 255).astype(np.uint8)
      #  residual = (residual / self.cfg.RESIDUAL_NORMAL_SCALE - self.cfg.RESIDUAL_MEANS) / self.cfg.RESIDUAL_STDS
#        residual_pil = Image.fromarray(np.uint8(residual*255))
#        residual_pil.save(f'resd/residual{self.frame_id}.jpg')

        ori_img, framed_img, framed_meta = preprocess_video_frame_residual(residual, max_size=input_sizes[self.compound_coef_resd], mean=params['mean'], std=params['std'])  #preprocess + rescale

        print('ori_img',ori_img.shape)

        original_height, original_width = ori_img.shape[:2]
        print(original_height, original_width)       #(562, 999)
        resized_height, resized_width = framed_img.shape[:2]
        print(resized_height, resized_width)  # 768 768 pour eff2

        x = torch.from_numpy(framed_img)

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        print('x',x.shape)
       # print('x',x)
         # model predict
        with torch.no_grad():   # desactive la backpropagation
            features, regression, classification, anchors = self.resd_model(x)   # FORWARD INFERENCE

        out = postprocess(x, anchors, regression, classification,regressBoxes, clipBoxes,
                          self.threshold, self.iou_threshold)  #iou_threshold = nms_ threshold

        #detector result
        out = invert_affine([framed_meta], out)[0]

        bbox_xyxy = out['rois']    # extrait les coordonnees des BB tlbr
        print('bbox_xyxy NON KEY',bbox_xyxy.shape)

        if len(bbox_xyxy.shape) == 1 and bbox_xyxy.size == 4:
            bbox_xyxy = bbox_xyxy.reshape(1, -1)

        if bbox_xyxy.size > 0:
            bbox_xywh = self.xyxy_to_xywh(bbox_xyxy)    # x_center, y_center, width, height
           # print("bbox_xywh:", bbox_xywh)
        else:
            bbox_xywh = np.empty(shape=(0, 4))


            #print("bbox_xywh.shape:", len(bbox_xywh.shape))

        cls_ids = out['class_ids']   # les identifiants des classes
        cls_conf = out['scores']      # les scores de confiance

        self.image_scale = torch.tensor([
            float(round(resized_width)) / float(original_width),   #768/999 (eff2)
            float(round(resized_height)) / float(original_height), #768/562 (eff2)
            float(round(resized_width)) / float(original_width),
            float(round(resized_height)) / float(original_height),
        ], dtype=torch.float32).cuda()
        print('self.image_scale non KEY',self.image_scale)
       # bbox_xywh*=self.image_scale       #modifie   dans le plan 768  768



        def process_outputs( cls_ids, bbox_xywh, cls_conf, im_data_tmp):
            tracker_out = {'rois': [], 'class_ids': [], 'obj_ids': [], 'scores': []}

            for index, target in enumerate(self.selected_target):
                mask = cls_ids == target
                bbox = bbox_xywh[mask]

                conf = cls_conf[mask]
                outputs = self.trackers[index].update(bbox, conf, im_data_tmp)

                if len(outputs) > 0:
                    confidence = np.full((outputs.shape[0], 1), 1)
                    outputs_with_conf = np.column_stack((outputs[:, 0:4]*self.image_scale, confidence))

                    tracker_out['rois'].extend(outputs_with_conf[:, :4])
                    tracker_out['class_ids'].extend(np.repeat(target, outputs.shape[0]))
                    tracker_out['obj_ids'].extend(outputs[:, -1])
                    tracker_out['scores'].extend(outputs_with_conf[:, 4])

            return tracker_out

        def process_tracking_results(tracker_out, residual_scale):
            for j in range(len(tracker_out['rois'])):
                xyxy = tracker_out['rois'][j].astype(int)
                tlwh = self.xyxy_to_tlwh(xyxy)
                tlwh = torch.tensor(tlwh, dtype=torch.float32, device='cuda') / self.image_scale       #modifie   revenir au plan 562 999
                tlwh = [x / residual_scale for x in tlwh]  # REVENIR AU PLAN 1080 1920


                obj_id = int(tracker_out['obj_ids'][j])
                one_data = torch.tensor([self.frame_id, obj_id, *tlwh, 1, -1, -1, -1], dtype=torch.float, device='cuda').unsqueeze(dim=0)
                if self.tracking_results is None:
                    self.tracking_results = one_data
                else:
#               print('self.tracking_results',self.tracking_results)
#                print('one_data',one_data)
                    self.tracking_results = torch.cat((self.tracking_results, one_data), dim=0)
#            print('self.tracking_results',self.tracking_results)


        def save_tracks( det_bboxes, obj_ids, residual_scale):
            detections_to_save = []
            detections_to_track = []

            if det_bboxes is not None:
                det_bboxes = det_bboxes.squeeze(dim=0)
                if det_bboxes.nelement() > 0:

                #    croped_f = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4]/residual_scale, in_data=self.key_features, scale=self.feature_scale , in_data_type='feature')
                    croped_f = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=self.key_features, scale=self.feature_scale , in_data_type='feature')

                if self.frame_id  % self.args.detection_interval ==0:
                    self.tracks=[]
                for i in range(det_bboxes.size()[0]):

                    #one_bbox = det_bboxes[i]/residual_scale  # [x1, y1, x2, y2, score]
                    one_bbox = det_bboxes[i]    # modife   rester dans le plan 768 768
                    one_f = None if croped_f is None else croped_f[i]
                    one_detection = Detection(tlbr=one_bbox[0:4], confidence=one_bbox[4], feature=one_f)
                    detections_to_save.append(one_detection)
                  #  print('det_bboxes ERRUER IDS', one_bbox[0:4])
                    if one_detection.confidence >= self.tracking_thr['DEEPSORT']:
                        if self.frame_id  % self.args.detection_interval ==0:
                            # last non key frame in a GOP
                            one_track = Track(one_detection, obj_ids[i], self.detector_name)
                            self.tracks.append(one_track)
                        else:
                            # first non key frame in a GOP
                            detections_to_track.append(one_detection)

            self.detections_to_save, self.detections_to_track = detections_to_save, detections_to_track
            if self.frame_id  % self.args.detection_interval ==2:
               print('self.frame_id  FIRST SPARSE, ASSOCIATION DONC', self.frame_id)
               self.associated_targets_detections()


#        tracker_out = process_outputs(cls_ids, bbox_xywh, cls_conf, im_data_tmp)  #plan 768  768

        tracker_out = process_outputs(cls_ids, bbox_xywh, cls_conf, ori_img)  #plan 768  768
        process_tracking_results(tracker_out, im_scale_tmp)
        det_bboxes = np.column_stack((tracker_out['rois'], tracker_out['scores']))  # (XYXY)
        det_bboxes = torch.from_numpy(det_bboxes)

        if det_bboxes.nelement() != 0:
            det_bboxes = det_bboxes.unsqueeze(0)

        if self.frame_id  % self.args.detection_interval ==0 or self.frame_id  % self.args.detection_interval == 2:   #first + last non-key frame in a GOP
            det_bboxes = det_bboxes.cuda().float()
            save_tracks(det_bboxes, tracker_out['obj_ids'], im_scale_tmp)



    def _detection_bbox_to_detection_list(self, det_bboxes, feature_map, f_scale):
        """
        convertir les b.boxes de détection (det_bboxes) en une liste de détections à enregistrer et à suivre. 
        Les détections sont créées à partir de b.boxes, de F.Maps, d'images, de mouvements et de résidus, si disponibles. 
        Les détections sont ensuite utilisées pour le suivi des objets.

        This function obtain the box from the output of the cnn_model. Noted that
        the returned boxes are not clipped and they are the coordinates of the
        testing image. So if you want the coordinates of the origin image, you need
        to clip and resize the boxes.

        :param det_bboxes: 3D tensor with size [bs, num_box, 5] : batch size , nb de b.boxes, 5= (x1, y1, x2, y2) et le score de confiance
        :param feature_map: 4D tensor, with size [bs, h, w ,c]
        :param f_scale: 1D tensor, with size [4]. The scales that
               obtained based on the size of im_data and featurer, [4]
               used to map rois from im_data to feature map
        :return: a list, each element in it is a 1D tensor, (x1, y1 ,x2 ,y2, score) : coins supérieur gauche et inférieur droit 
        """
        if not det_bboxes.is_cuda:
           # print("det_bboxes is not on CUDA. Moving it to CUDA.")
            det_bboxes = det_bboxes.cuda()
        if det_bboxes.dtype is not torch.float32:
           # print("det_bboxes is not of type FloatTensor. Converting it to FloatTensor.")
            det_bboxes = det_bboxes.float()

        f_scale = torch.tensor(f_scale, dtype=torch.float).cuda()

        # if there is detections for person
        detections_to_save = []
        detections_to_track = []

        if det_bboxes is not None:
            det_bboxes = det_bboxes.squeeze(dim=0) # remove the batch dim
       #     print('det_bboxes3',det_bboxes.shape)
       #     print('det_bboxes',det_bboxes)
            croped_im, croped_mv, croped_res, croped_f = None, None, None, None

            if self.args.iou_or_appearance in ['both', 'appearance', 'iou']:
                # feature map [bs, channels, h, w]. Noted that h, w here is not those in mv
                if det_bboxes.nelement() > 0 and feature_map is not None:
                     #   print('det_bboxes ERROR',det_bboxes)
                        croped_f = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=feature_map,
                                                     scale=f_scale, in_data_type='feature')
                else:
                        pass


            boxes = []
            confidences = []
            for i in range(det_bboxes.size()[0]):
                one_bbox = det_bboxes[i]  # [x1, y1, x2, y2, score]

                if one_bbox[4] >= self.tracking_thr[self.detector_name] and self.frame_id ==1:
                    bbox_np = one_bbox.cpu().numpy()
                    boxes.append(bbox_np)  # Use the numpy version here
                    confidences.append(1)

                one_im = None if croped_im is None else croped_im[i]  # [2, h, w]
 #               print(f"croped_im {i}: {one_im}")

                one_mv = None if croped_mv is None else croped_mv[i]  # [2, h, w]
  #              print(f"croped_mv {i}: {one_mv}")

                one_res = None if croped_res is None else croped_res[i]
#                print(f"croped_res {i}: {one_res}")

                one_f = None if croped_f is None else croped_f[i]  # [c, h, w]
 #               print(f"croped_f {i}: {one_f}")
 #               print('self frameID',self.frame_id)
 #               print('one_bbox[0:4]',one_bbox[0:4])
 #               print('one_bbox[4]',one_bbox[4])
                one_detection = Detection(tlbr=one_bbox[0:4],
                                          confidence=one_bbox[4],
                                          feature=one_f,
                                          mv=one_mv,
                                          im=one_im,
                                          residual=one_res)
           #     print(f"detection {i}: {one_detection}")

                detections_to_save.append(one_detection)

                if one_detection.confidence >= self.tracking_thr[self.detector_name]:
            #        print("Detection added to detections_to_track")
                    detections_to_track.append(one_detection)
             #       print('one_detection',one_detection)
               # else:
               #     print("Detection not added to detections_to_track due to low confidence")

#            print("Total number of detections to track: ", len(detections_to_track))

#        print('detections_to_save',detections_to_save)
#        print('detections_to_track',detections_to_track)


        return detections_to_save, detections_to_track

 #   @profile
    def do_tracking(self):
        """
       Fonction de suivi pour toutes les trajectoires dans self.tracks en gardant le meme ID , ( Appelee si detection_interval>1)
        """

        # get the boxes in last frame
        if len(self.tracks) > 0:
            boxes = None
            for t in self.tracks:
                one_box = t.to_tlbr()
                # rescale to origin image
                # one_box = one_box / im_scale
                one_box = one_box.unsqueeze(dim=0)  # [1, 4], [x1, y1, x2, y2]
                if boxes is None:
                    boxes = one_box.new().resize_(0, 4)

                boxes=boxes.cuda()
                one_box=one_box.cuda()
                boxes = torch.cat((boxes, one_box), dim=0)  # [num_tracks, 4]

            boxes = boxes.unsqueeze(dim=0)  # [bs ,num_track, 4], here bs = 1
            self.boxes.data.resize_(boxes.size()).copy_(boxes).contiguous()

            # load the im_data
            frame_type = 0 if int((self.frame_id - 1) % self.gop_size) == 0 else 1
            t1 = time.time()
            im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file,
                                                                       frame_id=self.frame_id,
                                                                       load_data_for='track')

            # Diviser les données en mv et résiduels
       #     mv = im_data_tmp[:, :, 0:2]
        #    residual = im_data_tmp[:, :, 2:5]
         #   residual = (residual * self.cfg.RESIDUAL_STDS + self.cfg.RESIDUAL_MEANS) * self.cfg.RESIDUAL_NORMAL_SCALE
            # Dénormaliser mv
 #           mv = (mv * self.cfg.MV_STDS + self.cfg.MV_MEANS) * self.cfg.MV_NORMAL_SCALE

            # Dénormaliser résiduel
            # Enregistrer les données dénormalisées
#            np.save(f'images/mv/mv_{self.frame_id}.npy', mv)

#            residual_pil = Image.fromarray(np.uint8(residual*255))

            # Cr  er le dossier si'il n'exsite pas
 #           directory = f'images/residual/{self.seq}'
  #          if not os.path.exists(directory):
   #             os.makedirs(directory)

    #        # Enregistrer l'image r  siduelle
     #       filename = str(self.frame_id).zfill(6)
      #      residual_pil.save(f'{directory}/{filename}.jpg')

            load_time = time.time() - t1

            im_info_tmp = np.array([[im_data_tmp.shape[0], im_data_tmp.shape[1], im_scale_tmp, frame_type]],
                                   dtype=np.float32)
            im_info_tmp = torch.from_numpy(im_info_tmp)
            im_data_tmp = np.array(im_data_tmp[np.newaxis, :, :, :], dtype=np.float32)  # [bs, h, w, c]
            im_data_tmp = torch.from_numpy(im_data_tmp).permute(0, 3, 1, 2).contiguous()  # [bs, c, h, w]

            self.im_info.data.resize_(im_info_tmp.size()).copy_(im_info_tmp).contiguous()
            self.im_data.data.resize_(im_data_tmp.size()).copy_(im_data_tmp).contiguous()

            t2 = time.time()
            with torch.no_grad():
                output = self.tracking_model(self.boxes, self.im_data)  # the deltas, [bs, num_box, 4]
            track_time = time.time() - t2

            return output, load_time, track_time
        else:
            return  None, 0, 0

    def track_output_to_offsets(self, output):
        """
        This function perform tracking based on the regression deltas.
        :param output: the output of tracking net, which is 3D tensor (or Variable).
                    which has the size of [bs, num_box, 4]. Here, bs is 1, num_box
                    is the number of non-deleted tracks in self.tracks.
        :return:
        """
        # obtain bounding-box regression deltas
        if output is not None:

            if isinstance(output, Variable):
                output = output.data

            box_deltas = output.clone()
            batch_size = box_deltas.size(0)
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * self.bbox_reg_std + self.bbox_reg_mean  # [num_box, 4]
                box_deltas = box_deltas.view(batch_size, -1, 4)
            boxes = bbox_transform_inv(boxes=self.boxes.data, deltas=box_deltas,
                                       sigma=self.tracking_model.transform_sigma)  # [1, num_box, 4]

        #    print('boxes',boxes.shape)
        #    print('boxes',type(boxes))
        #    print('len(self.tracks)',len(self.tracks))

            for t_idx in range(len(self.tracks)):
                self.tracks[t_idx].tracking(bbox_tlbr=boxes[0, t_idx, :])

    def initiate_track(self, detection):
        """
        This function add a track Tracker
        :param detection: detection.Detection
        :return: no return
        """
        self._next_id += 1
        one_track = Track(detection, self._next_id, self.detector_name)
        self.tracks.append(one_track)

    def _match_iou(self, track_candidates, detection_candidates):
        """
        This function match the tracks with detections based on the iou
        :param track_candidates: list, the index of tracks in self.tracks
        :param detection_candidates: list, the index of detections in self.detections_to_track
        :return:
        """
        if len(track_candidates) == 0 or len(detection_candidates) == 0:
            matches = []
            unmatched_tracks = track_candidates
            unmatched_detections = detection_candidates
        else:
            cost_matrix = distance_metric_func.iou_cost(tracks=self.tracks,    #  #  la matrice contient l'IoU pour chaque paire de detection
                                                        detections=self.detections_to_track,
                                                        track_indices=track_candidates,
                                                        detection_indices=detection_candidates)
          #  print('iou_cost_matrix',cost_matrix)
            # associate the tracks with the detections using iou
           # print('self.tracks',self.tracks)
           # print('self.detections_to_track',self.detections_to_track)
           # print('track_indices',track_candidates)
           # print('detection_candidates',detection_candidates)
            #  # trouver les correspondances de cout minimal (meilleur IoU) entre les trajectoires et les detections
            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching_v2(max_distance=self.max_iou_distance,
                                                       tracks=self.tracks,
                                                       detections=self.detections_to_track,
                                                       track_indices=track_candidates,
                                                       detection_indices=detection_candidates,
                                                       cost_matrix=cost_matrix,
                                                       cost_type='iou')

        return matches, unmatched_tracks, unmatched_detections      # # les correspondances entre trajectoires et detections, les trajectoires + detections non matchees

    def _match_appearance(self, track_candidates, detection_candidates):
        """
        This function match the tracks with detections based on appearance features
        :param track_candidates: list, the index of tracks in self.tracks
        :param detection_candidates: list, the index of detections in self.detections_to_track
        :return:
        """

        if len(track_candidates) == 0 or len(detection_candidates) == 0:
            matches = []
            unmatched_tracks = track_candidates
            unmatched_detections = detection_candidates
            print('len(track_candidates)',len(track_candidates))
            print('len(detection_candidates)',len(detection_candidates))
        else:

            # compute the euclidean distances between boxes
            #print('self.im_info2',self.im_info)
           # # matrice de cout basee sur la distance Eucld entre les boxes des pistes et detections

            dist_cost_matrix = distance_metric_func.euclidean_distance_cost(tracks=self.tracks,
                                                                            detections=self.detections_to_track,
                                                                            track_indices=track_candidates,
                                                                            detection_indices=detection_candidates,
                                                                            image_size=self.image_size)

            print('dist_cost_matrix',dist_cost_matrix)

            # # vrai la ou la distance est inferieure a max_euclidean_distance ,identifier les trajectoires et detections qui sont proches : appariement possible
            mask = dist_cost_matrix <= self.max_euclidean_distance
           # un tableau 2D de paires d'indices, chaque paire = un track_candidate detection_candidate
           # (qui sont  suffisamment proches en termes de distance euclidienne)
            print('mask',mask)
            index = torch.nonzero(mask)
            cost_matrix = self.cost_matrix[0:len(track_candidates), 0:len(detection_candidates)].clone()

            if index.size():
                num_pairs = index.size(0) # le nombre de paires track-detection
                print('num_pairs',num_pairs)
                # prepare features, stocker les features des objets suivis et des nouveaux détectés
                t_feature = None
                d_feature = None
                t_history = [] # used to store the history time of tracks

                for idx_p in range(num_pairs): # iterer sur chaque paire track-detection.
                    t_candidate_idx = track_candidates[index[idx_p, 0]]    #l'indiice d'objet existant
                    d_candidate_idx = detection_candidates[index[idx_p, 1]]  # l'indice de l'objet detecte en ce frame

                    t_feature_tmp = self.tracks[t_candidate_idx].feature  # [history, c, h, w]  # les features de l'objet suivi
                   # print('t_feature_tmp TEST',t_feature_tmp)
                    d_feature_tmp = self.detections_to_track[d_candidate_idx].feature.unsqueeze(dim=0)  # [1, c, h, w] # les features de l'objet dectecte
                    d_feature_tmp = d_feature_tmp.repeat(t_feature_tmp.size(0), 1, 1, 1)  # [history,c,h,w] repete d_feature_tmp pour avoir la meme dimension que t_feature_tmp

                    t_history.append(t_feature_tmp.size(0)) #  ajoute le nombre d'elements d'historique de l'objet suivi actuel
    #                print('idx_p',idx_p)

                    if idx_p == 0:   # Si premiere iteration
                        t_feature = t_feature_tmp
                        d_feature = d_feature_tmp
                    else:
                        t_feature = t_feature.cuda()
                        t_feature_tmp = t_feature_tmp.cuda()
                        d_feature = d_feature.cuda()
                        d_feature_tmp = d_feature_tmp.cuda()

                        t_feature = torch.cat((t_feature, t_feature_tmp), dim=0)
                        d_feature = torch.cat((d_feature, d_feature_tmp), dim=0)

                # forward to get the appearance similarities
  #              print('t_feature TEST ',t_feature.shape)
  #              print('d_feature TEST ',d_feature.shape)
                if t_feature is not None:
                    self.track_feature.data.resize_(t_feature.size()).copy_(t_feature).contiguous()
                    self.detection_feature.data.resize_(d_feature.size()).copy_(d_feature).contiguous()
                else:
                    print("t_feature is None, cannot resize or copy data.")


                # prob: [num_f, 2], vis_mask_t: [num_f, h*w, h, w], vis_mask_d: [num_f, h*w, h, w]
                # the memory is limited, so we need to divide to
                bs = self.track_feature.size(0)
                print('bs TEST: {}'.format(bs))
                max_bs = 100 #3000  # 10000
                num_bs = bs // max_bs   # le nombre de batchs qui seront traitees par le modele d'apparence.
                torch.cuda.empty_cache()

                if num_bs == 0:         # si le nombre de batchs est inférieur a max_bs.
                    t1 = time.time()
                    with torch.no_grad():
                        prob, vis_mask_t, vis_mask_d = self.appearance_model(self.track_feature, self.detection_feature)  #MODELE D'APPARENCE (CNN)
                    t2 = time.time()
                  #  print('sbc forward time: {}'.format(t2 - t1))
                else:
                    for i in range(num_bs + 1):   #  itere sur chaque batch
                        start = i * max_bs   # définir les indices de debut            pour le batch actuel
                        end = min((i+1)*max_bs, bs) #                       et de fin 

                        if start != end:     # batch n'est pas vide,  # retourner la probabilite de matching, les masques de visualisation pour les tracks et les drtections

                            with torch.no_grad():
                                prob_tmp, vis_mask_t_tmp, vis_mask_d_tmp = \
                                    self.appearance_model(self.track_feature[start:end, :, :, :],
                                                          self.detection_feature[start:end, :, :])

                            if i == 0:      # si premier batch
                                prob = prob_tmp.clone()    # stocker les probas directement
                                #vis_mask_t = vis_mask_t_tmp.clone()
                                #vis_mask_d = vis_mask_d_tmp.clone()
                            else:
                                prob = torch.cat((prob, prob_tmp), dim=0)    #  les nouveaux resultats sont concatenes aux resultats existents
                                #vis_mask_t = torch.cat((vis_mask_t, vis_mask_t_tmp), dim=0)
                                #vis_mask_d = torch.cat((vis_mask_d, vis_mask_d_tmp), dim=0)

                prob = prob.data # 2D, [num_f, 2]  # convertir le tensor prob en un numpy array
          #      print("prob TEST",prob)
#                prob = torch.flip(prob, [1])
#                print("prob TEST flipped ",prob)
                count = 0
                for idx_p in range(num_pairs):   # itrre sur chaque paire track/detection
                    count_next = count + t_history[idx_p]
                    one_prob = prob[count: count_next, 0] # # the probability of been different targets
                    one_prob, _ = one_prob.min(dim=0)
                    row = index[idx_p, 0]
                    col = index[idx_p, 1]
                    cost_matrix[row:row+1, col:col+1] = one_prob
                    count = count_next

            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching_v2(max_distance=self.max_appearance_distance,
                                                       tracks=self.tracks,
                                                       detections=self.detections_to_track,
                                                       track_indices=track_candidates,
                                                       detection_indices=detection_candidates,
                                                       cost_matrix=cost_matrix,
                                                       cost_type='appearance')
       # print('matches',matches)
       # print('unmatched_tracks',unmatched_tracks)
       # print('unmatched_detections',unmatched_detections)
        return matches, unmatched_tracks, unmatched_detections  #les matchs, les objets de suivi et détection non apparies



    def _match_iou_and_appearance(self, confirmed_tracks, unconfirmed_tracks, unmatched_detections, first_use=None):
        """
        This function match the tracks with detections based on appearance features and iou.
        :param confirmed_tracks: list, the index of confirmed tracks in self.tracks
        :param unconfirmed_tracks: the index of unconfirmed tracks in self.tracks
        :param unmatched_detections: list, the index of detections in self.detections_to_track
        :param first_use: str, 'iou', 'appearance' or 'joint'. The first cost type to use. If
                    'joint', the cost matrix is obtained by the weighted average of iou cost
                    matrix and appearance cost matrix.
        :return:
        """
        if first_use == 'iou':
            # first match the confirmed tracks with detections based on iou,
            # then match the unmatched tracks in confirmed tracks and unconfirmed
            # tracks with the unmatched detections base on appearance.
            track_candidates = confirmed_tracks
            detection_candidates = unmatched_detections
            # t1 = time.time()
            matches_iou, unmatched_tracks_iou, unmatched_detections_iou = \
                self._match_iou(track_candidates=track_candidates,
                                detection_candidates=detection_candidates)
            # t2 = time.time()
            # print('IOU match cost: {}'.format(t2-t1))

            # compute the similarity matrix
            track_candidates = unconfirmed_tracks + unmatched_tracks_iou
            detection_candidates = unmatched_detections_iou

            matches_app, unmatched_tracks_app, unmatched_detections_app = \
                self._match_appearance(track_candidates=track_candidates,
                                       detection_candidates=detection_candidates)
            matches = matches_app + matches_iou

            return matches, unmatched_tracks_app, unmatched_detections_app
        elif first_use == 'appearance':
            # first match the unconfirmed tracks with detections based on appearance,
            # then match the unmatched tracks in unconfirmed tracks and confirmed
            # tracks with the unmatched detections base on appearance.
            track_candidates = confirmed_tracks
            detection_candidates = unmatched_detections

            matches_app, unmatched_tracks_app, unmatched_detections_app = \
                self._match_appearance(track_candidates=track_candidates,
                                       detection_candidates=detection_candidates)

            track_candidates = unconfirmed_tracks + unmatched_tracks_app
            detection_candidates = unmatched_detections_app
            matches_iou, unmatched_tracks_iou, unmatched_detections_iou = \
                self._match_iou(track_candidates=track_candidates,
                                detection_candidates=detection_candidates)

            matches = matches_app + matches_iou

            return matches, unmatched_tracks_iou, unmatched_detections_iou
        elif first_use == 'joint':
            # match the all tracks with detections based on appearance and iou.
            # the cost matrix is obtained by the weighted average of iou cost
            # matrix and appearance cost matrix.

            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

            if len(track_candidates) == 0 or len(detection_candidates) == 0:
                matches = []
                unmatched_tracks = track_candidates
                unmatched_detections = detection_candidates
            else:
                # obtain the iou cost matrix
                iou_cost_matrix = distance_metric_func.iou_cost(tracks=self.tracks,
                                                                detections=self.detections_to_track,
                                                                track_indices=track_candidates,
                                                                detection_indices=detection_candidates)

                # obtain the appearance cost matrix
                # compute the euclidean distances between boxes
                dist_cost_matrix = distance_metric_func.euclidean_distance_cost(tracks=self.tracks,
                                                                                detections=self.detections_to_track,
                                                                                track_indices=track_candidates,
                                                                                detection_indices=detection_candidates,
                                                                                image_size=self.image_size)

                # finde the combinations that need to get the appearance similarity
                mask = dist_cost_matrix <= self.max_euclidean_distance
                # 2D tensor, the index of tracks and detection. The first column
                # if the index of tracks, and the second column is the index of
                # detections.  Noted that the index is based on
                # track_candiadtes and detection candidates
                index = torch.nonzero(mask)

                # app_cost_matrix = torch.zeros((len(track_candidates), len(detection_candidates))).float().fill_(
                #     linear_assignment.INFTY_COST)
                app_cost_matrix = self.cost_matrix[0:len(track_candidates), 0:len(detection_candidates)].clone()
                if index.size():
                    num_pairs = index.size(0)

                    # prepare features
                    t_feature = None
                    d_feature = None
                    t_history = []  # used to store the history time of tracks
                    for idx_p in range(num_pairs):
                        t_candidate_idx = track_candidates[index[idx_p, 0]]
                        d_candidate_idx = detection_candidates[index[idx_p, 1]]

                        t_feature_tmp = self.tracks[t_candidate_idx].feature  # [num_f, c, h, w]
                        d_feature_tmp = self.detections_to_track[d_candidate_idx].feature.unsqueeze(
                            dim=0)  # [1, c, h, w]
                        d_feature_tmp = d_feature_tmp.repeat(t_feature_tmp.size(0), 1, 1, 1)  # [num_f, c, h, w]

                        t_history.append(t_feature_tmp.size(0))

                        if idx_p == 0:
                            t_feature = t_feature_tmp
                            d_feature = d_feature_tmp
                        else:
                            t_feature = torch.cat((t_feature, t_feature_tmp), dim=0)
                            d_feature = torch.cat((d_feature, d_feature_tmp), dim=0)

                    # forward to get the appearance similarities
                    self.track_feature.data.resize_(t_feature.size()).copy_(t_feature).contiguous()
                    self.detection_feature.data.resize_(d_feature.size()).copy_(d_feature).contiguous()

                    # prob: [num_f, 2], vis_mask_t: [num_f, 1, h, w], vis_mask_d: [num_f, 1, h, w]
                    prob, vis_mask_t, vis_mask_d = self.appearance_model(self.track_feature, self.detection_feature)

                    prob = prob.data  # 2D, [num_f, 2]
                    count = 0

                    for idx_p in range(num_pairs):
                        count_next = count + t_history[idx_p]
                        one_prob = prob[count: count_next, 0]  # # the probability of been different targets
                        one_prob, _ = one_prob.min(dim=0)
                        row = index[idx_p, 0]
                        col = index[idx_p, 1]
                        app_cost_matrix[row:row + 1, col:col + 1] = one_prob
                        count = count_next

                iou_cost_weight = 0.5
                if not iou_cost_matrix.is_cuda and self.args.cuda:
                    iou_cost_matrix = iou_cost_matrix.cuda()
                cost_matrix = iou_cost_weight * iou_cost_matrix + (1 - iou_cost_weight) * app_cost_matrix
                max_distance = iou_cost_weight * self.max_iou_distance + (1 - iou_cost_weight) * self.max_appearance_distance

                matches, unmatched_tracks, unmatched_detections = \
                    linear_assignment.min_cost_matching_v2(max_distance=max_distance,
                                                           tracks=self.tracks,
                                                           detections=self.detections_to_track,
                                                           track_indices=track_candidates,
                                                           detection_indices=detection_candidates,
                                                           cost_matrix=cost_matrix,
                                                           cost_type='joint')

            return matches, unmatched_tracks, unmatched_detections
        else:
            raise RuntimeError('Unknown type of fisrt use: {}'.format(first_use))



    def match(self,cost_nonkey=None):
        """
        This function match detections with the tracks.
        :return:
        """

        confirmed_tracks = []
        unconfirmed_tracks = []

        for i, t in enumerate(self.tracks):
           # print('t.is_confirmed(): test ',t.is_confirmed())
            if t.is_confirmed():
                confirmed_tracks.append(i)
            elif t.is_tentative():
                unconfirmed_tracks.append(i)
#        print('confirmed_tracks',confirmed_tracks)
#        print('unconfirmed_tracks.append',unconfirmed_tracks)
        unmatched_detections = list(range(len(self.detections_to_track)))
       # print('unmatched_detections MATCH',unmatched_detections)

        if self.args.iou_or_appearance == 'iou' or cost_nonkey=='iou':
            print(' ASSOCIATION SPARSE')
            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

  #          print('track_candidates MATCH',track_candidates)
 #           print('detection_candidates MATCH',detection_candidates)

            return self._match_iou(track_candidates=track_candidates, detection_candidates=detection_candidates)

        elif self.args.iou_or_appearance == 'appearance':

            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

            return self._match_appearance(track_candidates=track_candidates,
                                          detection_candidates=detection_candidates)

        elif self.args.iou_or_appearance == 'both':

            return self._match_iou_and_appearance(unconfirmed_tracks=unconfirmed_tracks,
                                                  confirmed_tracks=confirmed_tracks,
                                                  unmatched_detections=unmatched_detections,
#                                                  first_use='joint')
                                                  first_use='iou')

    def associated_targets_detections(self, cost_nonkey = None):
        """
        This function associate the detections with the tracks
        :return: no return
        """
        matches, unmatched_tracks, unmatched_detections = self.match(cost_nonkey)
        print('matches',matches)
        print('unmatched_tracks',unmatched_tracks)
        print('unmatched_detections',unmatched_detections)
        # update the track set
        for track_idx, detection_idx, cost, distance_type in matches:


            self.tracks[track_idx].update(detection=self.detections_to_track[detection_idx],
                                          cost=cost, distance_type=distance_type)

        #print('detections_to_track',self.detections_to_track)
       # print('detections_to_track shape',self.detections_to_track.shape)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].predict()

        # initiate tracks
        if self.frame_id == 1 and self._next_id != 0:
            raise ValueError('In the first frame, the number of tracks should be 0 before initialize tracks,'
                             ' but found {}.'.format(self._next_id))
        print('unmatched_detections INTIAITE ',unmatched_detections)
        for detection_idx in unmatched_detections:
            self.initiate_track(self.detections_to_track[detection_idx])

        # filter out those deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def save_tracking_and_detection_results(self):
        """
        This function write the current tracking results
        :param im_scale: scalar, the resized scale for the image
        :return:
        """

        boxes = []
        confidences = []

        for t in self.tracks:
            if t.is_confirmed():
                bbox = t.to_tlwh()
               # print('bboxDEBUGGGGGG',bbox.shape)
                confidence = t.confidence

                if self.tracking_results is None:
                    self.tracking_results = bbox.new().resize_(0, 10)

                self.image_scale = self.image_scale.cuda()
                bbox = bbox.cuda()   #modifie

                bbox = bbox /self.image_scale   #modifie
               # print('bboxAPRES',bbox.shape)
                one_data = bbox.new([self.frame_id, t.track_id, bbox[0], bbox[1], bbox[2], bbox[3], confidence, -1, -1, -1]).unsqueeze(dim=0)
                one_data = one_data.cuda()
                self.tracking_results = torch.cat((self.tracking_results, one_data), dim=0)
            #    boxes.append(bbox_np)  # Use the numpy version here
            #    confidences.append(confidence)

      #  print('self.tracking_results',self.tracking_results)
        # Maj avec les nouvelles BOITES et confiances

        for d in self.detections_to_save:
            bbox = d.to_tlwh()
            feature = d.feature
            len_f = 0
            if feature is not None and self.args.save_detections_with_feature:
                feature = feature.view(1, -1) # 2D
                len_f = feature.size(1)

            confidence = d.confidence
            if self.detection_results is None:
                self.detection_results = bbox.new().resize_(0, 10 + len_f)

            bbox= bbox.cuda()
            self.image_scale = self.image_scale.cuda()   #modifie

            bbox = bbox /self.image_scale   #modifie

            one_data = bbox.new([self.frame_id, -1, bbox[0], bbox[1], bbox[2], bbox[3], confidence, -1, -1, -1]).unsqueeze(dim=0)
            if feature is not None and self.args.save_detections_with_feature:
                one_data = torch.cat((one_data, feature), dim=1) # [1, 10 + len_f]
            self.detection_results = torch.cat((self.detection_results, one_data), dim=0)

    def check_tensor_devices(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass
#    @profile
    def track_on_video(self, video_file, tracking_output_file, detection_output_file):

        """
        Given the path of a video, we do tracking on this video.
        :param video_file: str, the path to a video need to track
        :param tracking_output_file: the file path to save tracking results
        :param detection_output_file: the file path to save detection results
        :return: None
        """
        # Vérifieer si le fichier vidéo existe
        if not os.path.exists(video_file):
            raise RuntimeError(video_file + ' does not exists')

        # Récupère des informations sur le dataset, la séquence et le détecteur à partir des paths de fichiers
        video_file_list = video_file.split('/')
        self.seq = video_file_list[-2]    #MOT17-08-FRCNN ........
        self.tracked_seqs.append(self.seq)
        self.dataset_year = video_file_list[-4] # MOT16, MOT17
        self.phase = video_file_list[-3]   #train, test
        self.video_file = video_file




        # infact the num_frames + 1 is the true number of frames in this video
        num_frames = coviar.get_num_frames(video_file) + 1 #compter combien d'images (frames) dans la vidéo
        num_gops = coviar.get_num_gops(video_file)    #nb de gops

        self.num_frames.append(num_frames)  # add the number of frames for this video

        if num_frames // self.gop_size > num_gops:
            raise RuntimeError('Something wrong with the raw video.\n'
                               ' Number of frames: {}, number of GPOs: {}, GOP_SIZE: {}'.
                               format(num_frames, num_gops, self.gop_size))



        # TODO: begin to tracking
#        num_frames=20

        for frame_id in range(1, num_frames+1 ):  # frame id starts from 1  : Boucle sur tous les frames de la vidéo

            self.frame_id = frame_id


            if (self.frame_id - 1) % self.args.detection_interval == 0:     #KEY FRAMES

                print('I-FRAME :  {}, number of frames: {}/{}, number of tracks: {}.'.format(video_file, frame_id, num_frames, len(self.tracks)))
                # do detection on this frame

                load_time, detect_time = self.eff_det_key()

                self.load_time.append(load_time)
                self.detect_time.append(detect_time)

                association_start = time.time()
                self.associated_targets_detections()   # association des detections avec les objets existants (trajectoires)
                associate_time = time.time() - association_start

                self.associate_time.append(associate_time)

                self.save_tracking_and_detection_results()

            else:                                                             #NON-KEY FRAMES 
                print('P-FRAME : {}, number of frames: {}/{}, number of tracks {}.'.format(video_file, frame_id, num_frames, len(self.tracks)))
                # do not detect on this frame, just move the boxes of each track to the next frame
          #      output, load_time, track_time = self.do_tracking()
                self.eff_det_nonkey()
          #      t1 = time.time()
          #      self.track_output_to_offsets(output)
          #      offset_time = time.time() - t1

          #      self.save_tracking_and_detection_results()

                # save time
          #      self.load_time.append(load_time)
          #      self.track_time.append(track_time)
          #      self.offset_time.append(offset_time)

#                self.check_tensor_devices()

            if self.args.vis:
                self.visualize_results(kind_of_boxes='tracking')

        # reset the tracker so it can track on the next video
        self.reset(tracking_output_file=tracking_output_file, detection_output_file=detection_output_file)






