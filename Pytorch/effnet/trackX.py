# from lib.utils.misc import set_visible_devices
GPU_IDS = [0, 1]
# set_visible_devices(GPU_IDS)

import argparse

# from lib.tracking.tracker_eff2 import Tracker
# from lib.tracking.tracker import Tracker
from tracker_X import Tracker
from lib.model.tracking_net.rfcn_tracking_single_branch import RFCN_tracking
from lib.model.sbc_net.spatial_binary_classifier import SBC
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
# from lib.model.sbc_net.spatial_binary_classifier import SBC
# from lib.tracking.tracker import Tracker
import os
import numpy as np

# from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

import torch
import torch.nn as nn
import pprint
import cv2

##efficientdet model imports
from backbone import EfficientDetBackbone
from datetime import datetime

# from eff.efficientdet.utils import BBoxTransform, ClipBoxes
# from eff.utils.utils import preprocess, invert_affine, postprocess, preprocess_video_frame, preprocess_video

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='X  model')
    parser.add_argument('--mot_dir', default='datasets',
                        help='training dataset', type=str)

    parser.add_argument('--cuda', default=False,
                        action='store_true')
    parser.add_argument('--mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--vis', default=False,
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--detection_interval', default=1,
                        help='the key frame scheduler', type=int)
    parser.add_argument('--iou_or_appearance', default='iou', choices=['iou', 'appearance', 'both'],
                        help='the cost used for tracking', type=str)
    parser.add_argument('--dataset_year', default='MOT17',
                        choices=['MOT16', 'MOT17'],
                        help='the dataset to tracking', type=str)
    parser.add_argument('--dataset_args', default='video',
                        choices=['annotations', 'test', 'train', 'video', 'yolo'],
                        help='the dataset to tracking', type=str)
    parser.add_argument('--video_format', default='RAW',
                        help='the format of the video to perfrom  the tracking', type=str)
    parser.add_argument('--dataset_path', default='/home/modesto/PycharmProjects/compressed_tracking/datasets',
                        help='the path to datasets', type=str)
    parser.add_argument('--stage', default='train',
                        choices=['val', 'test', 'train'],
                        help="the phase for this running", type=str)
    parser.add_argument('--format', default='yolo',
                        choices=['yolo', 'eff', 'coco'],
                        help="the phase for this running", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # if not (len(GPU_IDS) > 0 and torch.cuda.available()):
    if len(GPU_IDS) == 0:
        args.cuda = False

    if args.cuda and len(GPU_IDS) > 1 and args.mGPUs:
        args.mGPUs = True
    else:
        args.mGPUs = False

    print('args.mGPUs', args.mGPUs)

    dataset = ['MOT17']  # ['MOT16', 'MOT17']

    phase = ['train']  # ['test']

    # print(args.stage)
    # TODO: the following operations will take some time, set them to False if a faster tracker is required.
    args.save_detections_with_feature = False  # save the detections, along with the cropped features

    args.feature_crop_size = (560, 11, 11)  # appearance crop size (h, w)

    args.detection_sbc_model = './save/detection_sbc_101_4_1_9417.pth'

    args.tracking_net_data_type = 'mv_residual'

    # ----------------------------- Efficientdet version for key frames --------------------------------

    compound_coef_key = 2  # ce parametre represente la version de modele efficientdet des frames clés
    # Charger le modele
    model = EfficientDetBackbone(compound_coef=compound_coef_key, num_classes=1)

    state_dict = torch.load('weights/efficientdet-d2_156_27648.pth')
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("USING", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = False

    # ----------------------------- Efficientdet version for NON-KEY frames --------------------------------

    compound_coef_resd = 2  # ce parametre represente la version de modele efficientdet des frames clés
    # Charger le modele
    model_resd = EfficientDetBackbone(compound_coef=compound_coef_resd, num_classes=1)

    # state_dict = torch.load('/workspace/logs/mot/3/mot_pytorch_residual/efficientdet-d3_46_98500.pth')
    state_dict = torch.load('weights/efficientdet-d2_156_27648.pth')
    model_resd.load_state_dict(state_dict, strict=False)

    model_resd.eval()
    model_resd = model_resd.cuda()

    for param in model_resd.parameters():
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        model_resd = nn.DataParallel(model_resd)

    # ---------------------------- tracking model --------------------------------------------
    """ tracking_model = RFCN_tracking(base_net='resnet', num_layers=18, data_type='mv_residual',
                                   pretrained=False, transform_sigma=args.tracking_box_transform_sigma)"""
    tracking_model = RFCN_tracking(base_net='resnet', num_layers=18, data_type='mv_residual',
                                   pretrained=False)
    #print(tracking_model)
    tracking_model.create_architecture()
    tracking_model.set_train_and_test_configure(phase='test')
    # checkpoint = torch.load(tracking_model)
    # tracking_model.load_state_dict(checkpoint['model'], strict=True)

    # ------------------------- sbc model -------------------------------------
    appearance_model = SBC(input_c=args.feature_crop_size[0],  # 560 for eff2, 800 pour eff3
                           input_h=args.feature_crop_size[1],
                           input_w=args.feature_crop_size[2])
    appearance_model.eval()
    #    sbc_model = '/workspace/save/sbc_eff3/motchallenge/sbc_eff3_1_24.pkl'
    # sbc_model = '/workspace/save/sbc_eff2_21juillet/motchallenge/sbc_eff2_21juillet_1_50.pkl'
    # sbc_model = '/workspace/save/sbc_25juillet/motchallenge/sbc_25juillet_1_40_4999.pkl'
    # checkpoint = torch.load(sbc_model)
    # print("checkpoint epoch",checkpoint['epoch'])
    # appearance_model.load_state_dict(checkpoint['model'])

    if args.cuda:
        model = model.cuda()
        model_resd = model_resd.cuda()
        tracking_model = tracking_model.cuda()
        appearance_model = appearance_model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)
        model_resd = nn.DataParallel(model_resd)
        tracking_model = nn.DataParallel(tracking_model)
        appearance_model = nn.DataParallel(appearance_model)

    print('Called with args:')
    print(args)

    tracker = Tracker(base_net_model=model,
                      resd_model=model_resd,
                      tracking_model=tracking_model,
                      appearance_model=appearance_model,
                      args=args, cfg=cfg, compound_coef=compound_coef_key,
                      compound_coef_resd=compound_coef_resd)

    mot_seqs = {

        'MOT17': {
            'train': ['MOT17-04', 'MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13', 'MOT17-02', 'MOT17-09'],
            'test': ['MOT17-12', 'MOT17-14'],
        },
        'MOT20': {
            'train': ['MOT20-03', ],  # 'MOT17-04'], # , 'MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13'], # 'MOT-04'],
        },
    }

    mot_seqs_train = mot_seqs['MOT17']['train']
    mot_seqs_test = mot_seqs['MOT17']['test']

    mot_info = {
        'MOT17': {'split': ['test', 'train']},
        'MOT20': {'split': ['test', 'train']},
    }

    dataset = args.dataset_year if args.dataset_year is not None else ['MOT17', 'MOT20']
    phase = args.stage if args.stage is not None else ['test']
    # print(dataset)
    # print(args.stage)

    subset = args.stage
    # print('tracking on ' + subset + ' using ' + ' X MODEL')
    # video_file = os.path.join(args.mot_dir, dataset ,"video","MOT17-02-DPM" + '.mp4')
    video_file = "/home/modesto/PycharmProjects/compressed_tracking/datasets/MOT17/video/MOT17-02-DPM-RAW.avi"
    # print('video_file', video_file)
    if not os.path.exists(video_file):
        raise RuntimeError(video_file + ' does not exists')

    if dataset == 'MOT17' or dataset == 'MOT20':
        tracking_output_path = os.path.join('./track_results_PROTOTYPE_X/',
                                            str(args.detection_interval),
                                            args.iou_or_appearance, dataset, subset)
        detection_output_path = os.path.join('./detect_results_PROTOTYPE_X',
                                             str(args.detection_interval),
                                             args.iou_or_appearance, dataset, subset)
        if not os.path.exists(tracking_output_path):
            os.makedirs(tracking_output_path)
        if not os.path.exists(detection_output_path):
            os.makedirs(detection_output_path)

        time_file = os.path.join(tracking_output_path, 'time_analysis' + 'X' + '.log')
        tracking_output_file = os.path.join(tracking_output_path, subset + '.txt')
        detection_output_file = os.path.join(detection_output_path, subset + '.txt')

        """tracker.track_on_video(video_file=video_file,
                               tracking_output_file=tracking_output_file,
                               detection_output_file=detection_output_file,
                               detector_name="DPM"
                               )"""

    tracker.save_time(time_file)
    dataset = os.path.join(args.dataset_path, args.dataset_year,args.dataset_args)
    for video in os.listdir(dataset):
        if args.video_format in video:
            video_path = os.path.join(dataset,video)
            if not os.path.exists(video_path):
                raise RuntimeError(video_path + ' does not exists')

            det_name = (video.split("-"))[2]


            tracking_output_path = os.path.join('./track_results_PROTOTYPE_X/',date,video)
            detection_output_path = os.path.join('./detect_results_PROTOTYPE_X',date ,video)
            if not os.path.exists(tracking_output_path):
                os.makedirs(tracking_output_path)
            if not os.path.exists(detection_output_path):
                os.makedirs(detection_output_path)

            time_file = os.path.join(tracking_output_path, 'time_analysis' + 'X' + '.log')
            tracking_output_file = os.path.join(tracking_output_path,'.txt')
            detection_output_file = os.path.join(detection_output_path,'.txt')
            tracker.track_on_video(video_file=video_path,
                                   tracking_output_file=tracking_output_file,
                                   detection_output_file=detection_output_file,
                                   detector_name=det_name)
        tracker.save_time(time_file)

    quit()
    for one_dataset in dataset:
        for s in args.stage:
            if s not in mot_info[one_dataset]['split']:
                print('dataset {} does not has stage {}'.format(one_dataset, stage))
            continue
            print('s', s)
            subset = mot_seqs[one_dataset][s]
            print('subset', subset)
            for seq in subset:
                if seq in mot_seqs_train:
                    s = 'train'
                elif seq in mot_seqs_test:
                    s = 'test'

                    #    if one_dataset == 'MOT17' or one_dataset == 'MOT20':
                    #       seq = seq + '-' + det_name
                    det_name = "FRCNN"
                    print('tracking on ' + seq + ' using ' + ' X MODEL')

                    video_file = os.path.join(args.mot_dir, one_dataset, s, seq, seq + '-mpeg4-1.0.mp4')
                    print('video_file', video_file)
                    if not os.path.exists(video_file):
                        raise RuntimeError(video_file + ' does not exists')

                    if one_dataset == 'MOT17' or one_dataset == 'MOT20':
                        tracking_output_path = os.path.join('./track_results_PROTOTYPE_X/',
                                                            str(args.detection_interval),
                                                            args.iou_or_appearance, one_dataset, s)

                    detection_output_path = os.path.join('./detect_results_PROTOTYPE_X',
                                                         str(args.detection_interval),
                                                         args.iou_or_appearance, one_dataset, s)
                    if not os.path.exists(tracking_output_path):
                        os.makedirs(tracking_output_path)
                    if not os.path.exists(detection_output_path):
                        os.makedirs(detection_output_path)

                    time_file = os.path.join(tracking_output_path, 'time_analysis' + 'X' + '.log')
                    tracking_output_file = os.path.join(tracking_output_path, seq + '.txt')
                    detection_output_file = os.path.join(detection_output_path, seq + '.txt')

                    tracker.track_on_video(video_file=video_file,
                                           tracking_output_file=tracking_output_file,
                                           detection_output_file=detection_output_file,
                                           detector_name=det_name
                                           )

                tracker.save_time(time_file)
