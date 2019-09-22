import os
import glob
from PIL import Image
from resizeimage import resizeimage
import sys
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as etree
import xml.etree.cElementTree as ET
from yattag import Doc, indent
import shutil
import pandas as pd       
from google_images_download import google_images_download
from io import BytesIO
import numpy as np
import tensorflow as tf
import datetime
def size_and_name(root_dir,query,pypath):
    i = 1
    z = 1
    main_dir = root_dir+'/'+'downloads'+'/'+query
    for filename in glob.iglob(main_dir + '**/*.jpg', recursive=True):
        print(filename)
        im = Image.open(filename)
        im = im.convert('RGB')
        im.save(filename , 'JPEG', quality=90)
    for filename in glob.iglob(main_dir + '**/*.png', recursive=True):
        print(filename)
        im = Image.open(filename)
        im = im.convert('RGB')
        im.save(filename , 'JPEG', quality=90)
    for filename in os.listdir(main_dir):
        tst =query + str(i) +'.jpg'
        src =main_dir+'/'+filename
        tst =main_dir+'/'+tst
        os.rename(src, tst)
        i = i+1
    for filename in glob.iglob(main_dir + '**/*.jpg', recursive=True):
        class DeepLabModel(object):
            INPUT_TENSOR_NAME = 'ImageTensor:0'
            OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
            INPUT_SIZE = 513
            FROZEN_GRAPH_NAME = 'frozen_inference_graph'
            def __init__(self, tarball_path):
    
                self.graph = tf.Graph()

                graph_def = None
                graph_def = tf.GraphDef.FromString(open(pypath+"/PSCMR_Tensorflow_object_trainer/"+tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

                if graph_def is None:
                      raise RuntimeError('Cannot find inference graph in tar archive.')

                with self.graph.as_default():
                      tf.import_graph_def(graph_def, name='')

                self.sess = tf.Session(graph=self.graph)

            def run(self, image):
                  start = datetime.datetime.now()
                  width, height = image.size
                  resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
                  target_size = (int(resize_ratio * width), int(resize_ratio * height))
                  resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
                  batch_seg_map = self.sess.run(
                      self.OUTPUT_TENSOR_NAME,
                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
                  seg_map = batch_seg_map[0]
                  end = datetime.datetime.now()
                  diff = end - start
                  print("Time taken to evaluate segmentation is : " + str(diff))
                  return resized_image, seg_map

        def drawSegment(baseImg, matImg):
            width, height = baseImg.size
            dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
            for x in range(width):
                for y in range(height):
                    color = matImg[y,x]
                    (r,g,b) = baseImg.getpixel((x,y))
                    if color == 0:
                        dummyImg[y,x,3] = 0
                    else :
                        dummyImg[y,x] = [r,g,b,255]
            img = Image.fromarray(dummyImg)
            print(filename)
            img.mode == 'RGB'
            img = img.convert('RGB')
            imResize = img.resize((600,600), Image.ANTIALIAS)
            imResize.save(filename , 'JPEG', quality=90)
            #img.save(outputFilePath)
        
        print(filename)
        inputFilePath = filename
        outputFilePath = root_dir+"/"+query+str(i)+'.jpg'
        i = i + 1

        if inputFilePath is None or outputFilePath is None:
            print("Bad parameters. Please specify input file path and output file path")
            exit()

        modelType = "mobile_net_model"
        if len(sys.argv) > 3 and sys.argv[3] == "1":
            modelType = "xception_model"
        MODEL = DeepLabModel(modelType)
        print('model loaded successfully : ' + modelType)

        def run_visualization(filepath):
            try:
                print("Trying to open : " )
                jpeg_str = open(filepath, "rb").read()
                orignal_im = Image.open(BytesIO(jpeg_str))
            except IOError:
                print('Cannot retrieve image. Please check file: ' + filepath)
                return
    
            print('running deeplab on image %s...' % filepath)
            resized_im, seg_map = MODEL.run(orignal_im)
            drawSegment(resized_im, seg_map)
        run_visualization(inputFilePath)


