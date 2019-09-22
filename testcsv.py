import os
import sys
import glob
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as etree
import xml.etree.cElementTree as ET
from yattag import Doc, indent
import shutil
import pandas as pd



def testcsv(root_dir,query,test_images):
    test_images = int(test_images)
    i = 1
    main_dir = root_dir +'/'+'downloads'+'/'+query
    os.mkdir(main_dir+'/'+query+'test')
    dst1 = main_dir+'/'+query+'test'
    dst = root_dir+'/'+query+'test'
    for i in range(test_images):
        i = i+1
        file = main_dir+"/"+query+str(i)+".jpg"
        shutil.copy(file, dst1)
    lxml1 = 1
    for filename in os.listdir(dst1):
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

        name.text = dst1+'/'+filename

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
   
        tree.write(str(lxml1)+'.xml')
        lxml1 = lxml1 + 1
    os.mkdir(dst1+'/'+query+'xml')

    dest_dir = dst1+'/'+query+'xml'

    for file in glob.glob(root_dir+'/*.xml'):
        print(file)
        shutil.move(file,dest_dir)

#--------------------------CSV FILE SECTION--------------------------------------

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
        xml_df.to_csv('testlabels.csv', index=None)
        print('Successfully converted xml to csv.')
    
    os.mkdir(dst1+'/'+query+'CSV')
    csv = dst1+'/'+query+'CSV'
    for file in glob.glob(root_dir+'/*.csv'):
        print(file)
        shutil.move(file,csv)
        print('Done')
