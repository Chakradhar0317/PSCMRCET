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
