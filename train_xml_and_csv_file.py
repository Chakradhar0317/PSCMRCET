import os
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as etree
import xml.etree.cElementTree as ET
from yattag import Doc, indent
import shutil
import glob
import sys
import pandas as pd


def xml(root_dir,query):
    lxml = 1
    main_dir = root_dir +'/'+'downloads'+'/'+query
    for filename in os.listdir(main_dir):
        root = Element('annotation')
        tree = ElementTree(root)

        #enter foldername

        name = Element('folder')

        root.append(name)

        name.text = query

    #enter filename

        name = Element('filename')

        root.append(name)

        name.text = filename

    #enter path

        name = Element('path')

        root.append(name)

        name.text = main_dir+'/'+filename

    #going to source

        doc1 = ET.SubElement(root, "source")
        ET.SubElement(doc1,"database").text = "Unknown"

    #going to size

        doc2 = ET.SubElement(root, "size")
        ET.SubElement(doc2,"width").text = "600"
        ET.SubElement(doc2,"height").text = "600"
        ET.SubElement(doc2,"depth").text = "3"

    #going to segmented

        name = Element('segmented')

        root.append(name)

        name.text = '0'

    #going to object

        doc3 = ET.SubElement(root, "object")
        ET.SubElement(doc3,"name").text = query
        ET.SubElement(doc3,"pose").text = "Unspecified"
        ET.SubElement(doc3,"truncated").text = "0"
        ET.SubElement(doc3,"difficult").text = "0"
        doc4 = ET.SubElement(doc3,"bndbox")
        ET.SubElement(doc4,"xmin").text = "1"
        ET.SubElement(doc4,"ymin").text = "1"
        ET.SubElement(doc4,"xmax").text = "600"
        ET.SubElement(doc4,"ymax").text = "600"


        print(etree.tostring(root))
   
        tree.write(str(lxml)+'.xml')
        lxml = lxml+1
    os.mkdir(main_dir+'/'+query+'xml')

    dest_dir = main_dir+'/'+query+'xml'

    for file in glob.glob(root_dir+'/*.xml'):
        print(file)
        shutil.move(file,dest_dir)

    xml_list = []
 
    for xml_file in glob.glob(dest_dir+'/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
        xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv('trainlabels.csv', index=None)
        print('Successfully converted xml to csv.')
    
    os.mkdir(main_dir+'/'+query+'CSV')
    csv = main_dir+'/'+query+'CSV'
    for file in glob.glob(root_dir+'/*.csv'):
        print(file)
        shutil.move(file,csv)
        print('Done')

