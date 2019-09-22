import shutil
def shtrmv(pypath):
    shutil.rmtree(pypath+'/PSCMR_Tensorflow_object_trainer/training/faster_rcnn_inception_v2_coco_2018_01_28')
    shutil.rmtree(pypath+'/PSCMR_Tensorflow_object_trainer/training/ssd_inception_v2_coco_2018_01_28')
    shutil.rmtree(pypath+'/PSCMR_Tensorflow_object_trainer/training')
    print ("Remove done")

