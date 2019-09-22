import os
import shutil

def dir(pypath):
    os.mkdir(pypath+"/PSCMR_Tensorflow_object_trainer/training")
    src1 = pypath+"/PSCMR_Tensorflow_object_trainer/data_files/ssd_inception_v2_coco_2018_01_28"
    dst1= pypath+"/PSCMR_Tensorflow_object_trainer/training/ssd_inception_v2_coco_2018_01_28"
    src2 = pypath+"/PSCMR_Tensorflow_object_trainer/data_files/faster_rcnn_inception_v2_coco_2018_01_28"
    dst2 = pypath+"/PSCMR_Tensorflow_object_trainer/training/faster_rcnn_inception_v2_coco_2018_01_28"
    shutil.copytree(src1, dst1)
    shutil.copytree(src2, dst2)
    print("Done")
    
    
