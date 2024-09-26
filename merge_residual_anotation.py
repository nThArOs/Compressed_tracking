import os
import shutil
from distutils.dir_util import copy_tree

anotation_path = os.path.join(os.getcwd(), "MOT17")
residual_frame = os.path.join(os.getcwd(), "compressed_datasets/MOT17")
ouput_path = os.path.join(os.getcwd(), "datasets")
datasets_name = residual_frame.split("/")
datasets_name = datasets_name[-1]
output_path = os.path.join(ouput_path, datasets_name)

if not os.path.exists(output_path):
    os.makedirs(output_path)

print(os.listdir(anotation_path))


def get_anotation(anotation_path, folder_t, residual_frame_path, output_path):
    residual_frame_path = os.path.join(residual_frame_path, folder_t)
    anotation_path = os.path.join(anotation_path, folder_t)
    output_path = os.path.join(output_path, folder_t)
    annotation = ['gt', 'seqinfo.ini', 'det']
    frame = ['mv', 'mix-small', 'iframe', 'mix', 'residual']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for folder in os.listdir(anotation_path):
        folder_path = os.path.join(anotation_path, folder)
        output_folder_path = os.path.join(output_path, folder)
        frame_folder_path = os.path.join(residual_frame_path, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for annotation_item in annotation:
            output_anotation = os.path.join(output_folder_path, annotation_item)
            if os.path.exists(os.path.join(folder_path,annotation_item)):
                if annotation_item is not "seqinfo.ini":
                    if not os.path.exists(output_anotation):
                        os.makedirs(output_anotation)
                    copy_tree(os.path.join(folder_path, annotation_item),output_anotation)
                else:
                    shutil.copy(os.path.join(folder_path, annotation_item), output_folder_path)

        for frame_item in frame:
            output_frame = os.path.join(output_folder_path,frame_item)
            if not os.path.exists(output_frame):
                os.makedirs(output_frame)
            copy_tree(os.path.join(frame_folder_path, frame_item), output_frame)

get_anotation(anotation_path, "test", residual_frame, output_path)
