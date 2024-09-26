import os
import argparse
#dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

dirname = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--datasets', type=str, default='MOT17-residual', help='Name of the datasets')
    parser.add_argument('-f', '--folders', type=str, default=['train','test'], help='motion vector frame folder name')
    parser.add_argument('-g', '--gopsize', type=int, default='10', help='Size of the gop')
    parser.add_argument('-i', '--iframe', type=str, default='i-frame', help='i-frame folder name')
    parser.add_argument('-r', '--residual', type=str, default='img1', help='residual frame folder name')
    parser.add_argument('-mv', '--mv', type=str, default='mv', help='motion vector frame folder name')

    args = parser.parse_args()
    return args

arg = get_args()
datasets_path = os.path.join(dirname,'datasets/'+ arg.datasets)

for folder in arg.folders:
    folder_path = os.path.join(datasets_path, folder)
    os.listdir(folder_path)
    for vid in os.listdir(folder_path):
        vid_path = os.path.join(folder_path, vid)
        iframe_path = os.path.join(vid_path, arg.iframe)
        residual_path = os.path.join(vid_path, arg.residual)
        mv_path = os.path.join(vid_path, arg.mv)
        for frame in os.listdir(iframe_path):
            print(frame)
        #print(len(os.listdir(iframe_path)),len(os.listdir(residual_path)),len(os.listdir(mv_path)))
        #print(iframe_path,residual_path,mv_path)


#rename 's/\d+/sprintf("%05d",$&)/e' *.png