
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
from PIL import Image
import sys
import glob
import pandas as pd
import io
import tensorflow as tf
from PSCMR_Tensorflow_object_trainer.utils import dataset_util
from collections import namedtuple, OrderedDict

def test_tf_record(root_dir,query):
    main_dir = root_dir +'/'+'downloads'+'/'+query
    dst1 = main_dir+'/'+query+'test'
    def class_text_to_int(row_label):
        if row_label == query:
            return 1
        else:
            None


    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(group, path): 
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    os.mkdir(dst1+'/'+query+'testrecord')
    label_dir = dst1+'/'+query+'testrecord'
    writer = tf.python_io.TFRecordWriter(label_dir+'/'+'testrecord.record')
            
    examples = pd.read_csv(dst1+'/'+query+'CSV'+'/'+'testlabels.csv')
    grouped = split(examples, 'filename')
    for group in grouped:
            tf_example = create_tf_example(group, dst1)
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = dst1+'/'+query+'CSV'
    print('Successfully created the TFRecords: {}'.format(output_path))
    return(label_dir)
