import os
import subprocess
import configparser

folders = [
    #"MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN"
    #"MOT17-05-FRCNN",
    "residual"
]
dirname = os.path.dirname(__file__)
mot17_path = os.path.join(dirname, 'datasets/image')
output = os.path.join(dirname, 'datasets/video')
##mot17_path = os.path.join(dirname, 'datasets/image/MOT17/test')
##output = os.path.join(dirname, 'datasets/video/MOT17')

#mot17_path = "/home/jovyan/iadatasets/MOT/MOT17/test"
#output = "/home/jovyan/Desktop"

for folder in folders:
    img1_path = os.path.join(mot17_path, folder, "img1")
    output_path = os.path.join(output, f"{folder}-mpeg4.mp4")
    seqinfo_path = os.path.join(mot17_path, folder, "seqinfo.ini")

    if os.path.exists(img1_path) and os.path.exists(seqinfo_path):
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        frame_rate = config.getint("Sequence", "frameRate")

       # cmd = f"ffmpeg -y -f image2 -framerate {frame_rate} -i {img1_path}/%06d.jpg -c:v mpeg4 -f rawvideo -r {frame_rate} {output_path}"
        cmd = f"ffmpeg -y -f image2 -framerate {frame_rate} -i {img1_path}/%06d.jpg -c:v mpeg4 -r {frame_rate} {output_path}"

        subprocess.run(cmd, shell=True, check=True)
        print(f"Video created for {folder}")
    else:
        print(f"{img1_path} or {seqinfo_path} not found, skipping")
