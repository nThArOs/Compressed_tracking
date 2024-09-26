import cv2
from coviar import load
from coviar import get_num_gops
from coviar import get_num_frames
import os
from PIL import Image
import numpy as np
import subprocess
import torch

datasets_name = "MOT17"
input_path = os.path.join(os.getcwd(), datasets_name)

output_directory = "compressed_datasets"
output_path = os.path.join(os.getcwd(), output_directory)
output_path_datasetes = os.path.join(output_path, datasets_name)

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path_datasetes):
    os.makedirs(output_path_datasetes)

def get_frames_from_video(video, output_path):
    gop_index = 0
    frame_index = 0
    gop_size = 10
    imgid = 0
    num_gops = get_num_gops(video)
    num_frames = get_num_frames(video)
    print(num_gops,num_frames)
    create_img_directory(output_path)

    while(gop_index < num_gops):
        if gop_index+1 == num_gops:
            overly_frame = (num_gops * (gop_size+2)) - num_frames
            while (frame_index <= ((gop_size+2)-overly_frame)-1):
                print(f"Gop {gop_index} / {num_gops} / {imgid:06d}")
                save_image(video, gop_index, gop_size, frame_index, output_path, imgid)
                frame_index += 1
                imgid += 1
            gop_index += 1
        else:
            while (frame_index <= gop_size+1):

                print(f"Gop {gop_index} / {num_gops} / {imgid:06d}")
                save_image(video,gop_index,gop_size,frame_index,output_path,imgid)
                frame_index += 1
                imgid += 1
            if frame_index > gop_size:
                frame_index = 0
                gop_index += 1

def save_image(video, gop_index,gop_size,frame_index,output_path,imgid):
    tif = load(video, gop_index, frame_index, 0, True)  # Iframe
    tmv = load(video, gop_index, frame_index, 1, True)  # Motion vector frame
    tr = load(video, gop_index, frame_index, 2, True)  # residual frame

    tifRGB = cv2.cvtColor(tif, cv2.COLOR_BGR2RGB)
    imtif = Image.fromarray(tifRGB)  # Iframe image from the array
    imtmv = Image.fromarray((tmv * 255).astype(np.uint8))  # Mv image from the array
    imtr = Image.fromarray((tr * 255).astype(np.uint8))  # residual image from the array

    imtif.save(f"{os.path.join(output_path, 'iframe')}/{imgid:06d}.png")
    imtr.save(f"{os.path.join(output_path, 'residual')}/{imgid:06d}.png")
    imtmv.save(f"{os.path.join(output_path, 'mv')}/{imgid:06d}.png")

    if imgid % (gop_size + 2) == 0:
        imtif.save(f"{os.path.join(output_path, 'mix')}/{imgid:06d}-iframe.png")

        imtif.save(f"{os.path.join(output_path, 'mix-small')}/{imgid:06d}.png")
    else:
        imtr.save(f"{os.path.join(output_path, 'mix')}/{imgid:06d}-residual.png")
        imtmv.save(f"{os.path.join(output_path, 'mix')}/{imgid:06d}-mv.png")

        imtr.save(f"{os.path.join(output_path, 'mix-small')}/{imgid:06d}.png")
def create_img_directory(output_path):
    if not os.path.exists(os.path.join(output_path, "iframe")):
        os.makedirs(os.path.join(output_path, "iframe"))
    if not os.path.exists(os.path.join(output_path, "residual")):
        os.makedirs(os.path.join(output_path, "residual"))
    if not os.path.exists(os.path.join(output_path, "mv")):
        os.makedirs(os.path.join(output_path, "mv"))
    if not os.path.exists(os.path.join(output_path, "mix")):
        os.makedirs(os.path.join(output_path, "mix"))
    if not os.path.exists(os.path.join(output_path, "mix-small")):
        os.makedirs(os.path.join(output_path, "mix-small"))

def video_from_image_folder(folder_path, video_name, output_path):
    #ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' video.mp4

    if not os.path.isfile(os.path.join(output_path,video_name+'.mp4')):
        cmd_create_video = f"ffmpeg -framerate 25 -pattern_type glob -i '*.jpg' {output_path}/{video_name}.mp4"
        subprocess.run(cmd_create_video, shell=True, check=True, cwd=folder_path)
    if not os.path.isfile(os.path.join(output_path, video_name + '-RAW.avi')):
        cmd_create_raw_video = f"ffmpeg -i {output_path}/{video_name}.mp4 -vf setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo {output_path}/{video_name}-RAW.avi"
        subprocess.run(cmd_create_raw_video, shell=True, check=True, cwd=folder_path)

def get_mot_video(input_path):
    if "train" in os.listdir(input_path) and "test" in os.listdir(input_path): #check folder format for mot
        train_path = os.path.join(input_path, "train")
        test_path = os.path.join(input_path, "test")
        out_train_path = os.path.join(output_path_datasetes, "train")
        out_test_path = os.path.join(output_path_datasetes, "test")

        if not os.path.exists(out_train_path):
            os.makedirs(out_train_path)
        if not os.path.exists(out_test_path):
            os.makedirs(out_test_path)

        for folder in os.listdir(train_path):
            img_path = os.path.join(train_path, folder + "/img1")
            out_vid_path = os.path.join(out_train_path, folder)
            if not os.path.isdir(out_vid_path):
                os.makedirs(out_vid_path)
            video_from_image_folder(img_path,folder,out_train_path)

            video_path = os.path.join(out_train_path, folder+'-RAW.avi')
            output_video_path = os.path.join(out_train_path, folder)
            print(f"video_path{video_path}")
            print(f"output_video_path{output_video_path}")
            get_frames_from_video(video_path,output_video_path)

        for folder in os.listdir(test_path):
            img_path = os.path.join(test_path, folder + "/img1")
            out_vid_path = os.path.join(out_test_path, folder)
            if not os.path.isdir(out_vid_path):
                os.makedirs(out_vid_path)
            video_from_image_folder(img_path,folder,out_test_path)

            video_path = os.path.join(out_test_path, folder + '-RAW.avi')
            output_video_path = os.path.join(out_test_path, folder)
            print(f"video_path{video_path}")
            print(f"output_video_path{output_video_path}")
            get_frames_from_video(video_path, output_video_path)




    folders = os.listdir(input_path)

get_mot_video(input_path)
#get_frames_from_video("/home/modesto/PycharmProjects/compressed_tracking/compressed_datasets/MOT20/test/MOT20-04-RAW.avi","/home/modesto/PycharmProjects/compressed_tracking/compressed_datasets/MOT20/test/MOT20-04")
#get_frames_from_video(input_path)