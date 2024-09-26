
from timeit import default_timer as timer

import torch
import cv2
import numpy as np
from torch.backends import cudnn
import time
from backbone import EfficientDetBackbone
from deep_sort import build_tracker
from config import *

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video, xyxy_to_xywh, preprocess_video_residual

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class MOT(object):
    def __init__(self,
                 video_src: str,
                 video_output: str,
                 text_output: str,
                 obj_list: list,    # Liste des classes d'objets a suivre
                 input_sizes: list,   
                 reid_cpkt: str,     # path du fichier de poids du modele de reconnaissance d'objet re-id
                 compound_coef: int,   # Coefficient (version) du modele EfficientDet
                 force_input_size=None,
                 threshold=0.8,     # Seuil de confiance lors la detection des objets
                 iou_threshold=0.2,  #seil de NMS
                 use_cuda=True,
                 use_float16=False,
                 cudnn_fastest=True,
                 cudnn_benchmark=True,

                 max_dist=0,   # distance spatiale maximale pour associer les objets entre 2 frames successives   # SELON LES VITESSES PREVUES DES OBJETS 
                 max_iou_distance=0, # distance max pour associer 2 objets en termes d'iou
                 min_confidence=0,
                 nms_max_overlap=0,
                 
                 max_age=50,     #  nb maximal de frames pendant d'inactivité avant de supprimer une trajectoire
                 n_init=2,      # nb minimal de détections avant d'initialiser une nouvelle trajectoire, il faut qu'elle soit associee n_init fois consecutives avant de creer sa trajectoire
                 nn_budget=100,     # nb max de trajectoires a sauvegarder pour le calcul de la distance nearest neighbor

                 selected_target=None):   # liste des classes a suivre

        # I/O
        # Video's path
        self.seq = "mot-residual"
        self.video_src = os.path.join(dirname, f"MOT_VIDEOS/{self.seq}.mp4")
        self.video_output = os.path.join(dirname, f"MOT_VIDEOS/DeepSort_onMOT_residual/{self.seq}_DS.mp4")
        self.text_output = os.path.join(dirname, f"MOT_VIDEOS/DeepSort_onMOT_residual/{self.seq}.csv")
        #self.video_src = f"/home/jovyan/Desktop/MOT_VIDEOS/{self.seq}.mp4"
        #self.video_output = f"/home/jovyan/Desktop/MOT_VIDEOS/DeepSort_onMOT_residual/{self.seq}_DS.mp4"
        # text path
        #self.text_output =  f"/home/jovyan/Desktop/MOT_VIDEOS/DeepSort_onMOT_residual/{self.seq}.csv"

        # DETECTOR
        self.compound_coef = 2
        self.force_input_size = force_input_size  # set None to use default size

        self.threshold = 0.85
        self.iou_threshold = 0.3

        self.use_cuda = use_cuda
        self.use_float16 = False
        cudnn.fastest = cudnn_fastest
        cudnn.benchmark = cudnn_benchmark

        
        self.obj_list = obj_list

        # input size
        self.input_sizes = input_sizes
        self.input_size = input_sizes[self.compound_coef] if force_input_size is None else force_input_size

        # load detector model
        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=1)
        model.load_state_dict(torch.load(os.path.join(dirname, 'models_residual_trained/efficientdet-d2_49_26550.pth')))
        #model.load_state_dict(torch.load('/home/jovyan/Desktop/models_residual_trained/efficientdet-d2_49_26550.pth'))  # charger les poids pretrained , ceci est residual model
        model.requires_grad_(False)  # for evaluation
        model.eval()

        self.detector = model  
        if self.use_cuda and torch.cuda.is_available():
            self.detector = model.cuda()
            print('using CUDA')
        if self.use_float16:
            self.detector = model.half()
            print('use_float16')
            
        # TRACKER
        self.reid_cpkt = reid_cpkt
        self.max_dist = 0.17     # max_cosine_distance
        self.min_confidence = 0.3
        self.nms_max_overlap = 0.7
        self.max_iou_distance = 0.7
        self.max_age = 60
        self.n_init = 2
        self.nn_budget = 100

        # load tracker model,
        self.trackers = []
        self.selected_target = selected_target
        for num in range(0, len(self.selected_target)):
            self.trackers.append(build_tracker(reid_cpkt,
                                               self.max_dist,
                                               self.min_confidence,
                                               self.nms_max_overlap,
                                               self.max_iou_distance,
                                               self.max_age,
                                               self.n_init,
                                               nn_budget,
                                               use_cuda))
        # video frames
        self.frame_id = 1

    def xyxy_to_tlwh(self, bbox):
        # bbox: [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = bbox
        t = x1
        l = y1
        w = x2 - x1
        h = y2 - y1
        return [t, l, w, h]

    def _display(self, preds, imgs, text_recorder=None, track_result=None):  # afficher les resultats du tracking dans la video

        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

        if len(preds['rois']) == 0:
            return imgs

        # get color
        obj_ids = preds['obj_ids']
        u, indices = np.unique(obj_ids, return_inverse=True)

        # Open MOT17 format file to write into
        #mot17_file = open(f"/home/jovyan/Desktop/compare_eff_rfcn/{self.seq}.txt", "a")
        mot17_file = open(os.path.join(dirname, f"compare_eff_rfcn/{self.seq}.txt"), "a")
        last_results = None
        for j in range(len(preds['rois'])):
            # bbox
            xyxy = preds['rois'][j].astype(int)
            tlwh = self.xyxy_to_tlwh(xyxy)
            (t, l, w, h) = tlwh

            # info
            cls_id = self.obj_list[preds['class_ids'][j]]
            obj_id = int(preds['obj_ids'][j])
            # color
            color = [int((p * (obj_id ** 2 - obj_id + 1)) % 255) for p in palette]
            cv2.rectangle(imgs, (t, l), (t+w, l+h), color, 2)

            cv2.putText(imgs, '{}, {}'.format(cls_id, obj_id),
                        (t, l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

            # text recorded
            if text_recorder is not None:
                text_recorder.write(','.join([str(self.frame_id), str(obj_id), cls_id, str(t), str(l), str(w), str(h)]))
                text_recorder.write("\n")
                track_result[cls_id].add(obj_id)

            # Write to the MOT17 format file
            mot17_file.write(','.join([str(self.frame_id), str(obj_id), str(t), str(l), str(w), str(h), "1", "-1", "-1", "-1"]))
            mot17_file.write("\n")

        # Close the MOT17 format file
        mot17_file.close()
        print('self.frame_id',self.frame_id)
        self.frame_id +=1
        return imgs

    def detect_video(self):
        
        # Box postprocess
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # Video capture
        cap = cv2.VideoCapture(self.video_src)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))   # un identifiant pour specifier le codec video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   # la taille de la video en pixel
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # video recorder
        isVideoOutput = True if (self.video_output != "") or (self.video_output is not None) else False   # verifier si la sortie videp est specifiee
        if isVideoOutput:
            # print("TYPE:", type(self.video_output), type(video_FourCC), type(video_fps), type(video_size))
            output2video = cv2.VideoWriter(self.video_output, video_FourCC, video_fps, video_size)    # pour ecrire des images dans un fichier video self.video_output

        # text recorder
        isTextOutput = True if (self.text_output != "") or (self.text_output is not None) else False
        if isTextOutput:
            output2text = open(self.text_output, 'w', encoding='utf-8')
            output2text.write("Frame,Obj_ID,Type,x1,y1,x2,y2\n")    #ecirire l'entete dans le file CSV
            track_result = {}
            for obj_cls in self.obj_list:    # la liste des objets à suivre
                track_result[obj_cls] = set([])   # initialiser un dict pour stocker les resultats de tracking
        else:
            output2text = None
            track_result = None

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
      
        frame_count = 0
        start_time = time.time()
        total_track_time = 0 

        while True:
            ret, frame = cap.read()
            if self.frame_id % 12 ==1:
                self.frame_id += 1
                continue
           
            if not ret:
                break

            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video_residual(frame, max_size=self.input_size)  #pretrainer + rescale

            if self.use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)  # convertit les tensors en float32 ou float16 et rendre les canaux en 2eme position

            # model predict
            with torch.no_grad():   # desactive la backpropagation
            #    print('X', x.shape)
             #   print('X', x)
                
                features, regression, classification, anchors = self.detector(x)   # FORWARD INFERENCE

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  self.threshold, self.iou_threshold)  #iou_threshold = nms_ threshold

            # detector result
            out = invert_affine(framed_metas, out)
            # out = [{[xyxy], [class], [scores]}, ...]
            bbox_xyxy = out[0]['rois']    # extrait les coordonnees des BB tlbr
           # print('bbox_xyxy',bbox_xyxy.shape)
          #  print('bbox_xyxy',bbox_xyxy)

            if len(bbox_xyxy.shape) == 1 and bbox_xyxy.size == 4:
                bbox_xyxy = bbox_xyxy.reshape(1, -1)

            if bbox_xyxy.size > 0:
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)    # x_center, y_center, width, height
              #  print("bbox_xywh:", bbox_xywh)
            else:
                bbox_xywh = np.empty(shape=(0, 4))


            #print("bbox_xywh.shape:", len(bbox_xywh.shape))
            
            cls_ids = out[0]['class_ids']   # les identifiants des classes 
            cls_conf = out[0]['scores']      # les scores de confiance
            
            # tracker results
            # TRACKING
            tracker_out = {'rois': np.empty(shape=(0, 4)), 'class_ids': np.empty(shape=(0,), dtype=int),
                           'obj_ids': np.empty(shape=(0,), dtype=int)}
            
            frame_count += 1
            track_start_time = time.time()

           # print('bbox_xyxy',bbox_xyxy)
            for index, target in enumerate(self.selected_target):   # selectionner uniquement les classes d'interet
                mask = cls_ids == target
                bbox = bbox_xywh[mask]
                conf = cls_conf[mask]
               # print('bbox',bbox)
               # print('index, target',index, target)
               # print('bbox for update',bbox)
                outputs = self.trackers[index].update(bbox, conf, frame)  # MAJ le tracker avec les BBs et les confidences
              #  print("outputs for tracker {}: {}".format(index, outputs))

                if len(outputs) > 0:
                    tracker_out['rois'] = np.append(tracker_out['rois'], outputs[:, 0:4], axis=0)   #xyxy
                    tracker_out['class_ids'] = np.append(tracker_out['class_ids'], np.repeat(target, outputs.shape[0]))
                    tracker_out['obj_ids'] = np.append(tracker_out['obj_ids'], outputs[:, -1])

            
            track_end_time = time.time()
            total_track_time += track_end_time - track_start_time
            # show bbox info and results
            img_show = self._display(tracker_out, ori_imgs[0], output2text, track_result)
            
            # show frame by frame
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                fps = "FPS: " + str(int(curr_fps))
                curr_fps = 0
                accum_time = 0 # Réinitialiser accum_time

            # show FPS
            cv2.putText(img_show, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 255, 0), thickness=2)

            
       #     cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
       #     cv2.imshow("frame", img_show)
        #    cv2.waitKey(200)
            
            if isVideoOutput:
                output2video.write(img_show)
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        track_fps = frame_count / total_track_time
        print("Tracking FPS: ", track_fps)



        cap.release()
        cv2.destroyAllWindows()
        output2text.close()
        for obj_cls in self.obj_list:
            print(obj_cls + ': ' + str(len(track_result[obj_cls])))


if __name__ == "__main__":
    
    detector = MOT(
                   video_src,
                   video_output,
                   text_output,
                   obj_list,
                   input_sizes,
                   REID_CKPT,
                   compound_coef,
                   force_input_size,
                   threshold,
                   iou_threshold,
                   use_cuda,
                   use_float16,
                   cudnn_fastest,
                   cudnn_benchmark,

                   MAX_DIST,
                   MIN_CONFIDENCE,
                   NMS_MAX_OVERLAP,
                   MAX_IOU_DISTANCE,
                   MAX_AGE,
                   N_INIT,
                   NN_BUDGET,

                   selected_target
    )

    detector.detect_video()

