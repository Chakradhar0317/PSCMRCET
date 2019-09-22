from PSCMR_Tensorflow_object_trainer import images
from PSCMR_Tensorflow_object_trainer import size_name
from PSCMR_Tensorflow_object_trainer import train_xml_and_csv_file
from PSCMR_Tensorflow_object_trainer import train_tf_record
from PSCMR_Tensorflow_object_trainer import testcsv
from PSCMR_Tensorflow_object_trainer import test_tf_record
from PSCMR_Tensorflow_object_trainer import label_config
from PSCMR_Tensorflow_object_trainer import train
from PSCMR_Tensorflow_object_trainer import shtlrmv
from PSCMR_Tensorflow_object_trainer import datadir

#from PSCMR_Tensorflow_object_trainer import background_removal
from PSCMR_Tensorflow_object_trainer import export_inference_graph
def arrguments(query,jpg,png,root_dir,pypath,mode,num_steps,test_images,image_type):
    print("\n\n\n'POTTI SRI RAMULU CHALAVADI MALLIKARJUNARAO COLLLEGE OF ENGINEERING AND TECHNOLOGY'\n\nTRAINED BY FACULTY : MURALI KRISHNA; PRADEEP KUMAR \n\nSTUDENTS DONE  : DEDEEPYA KARA; APARNA MARRIVADA \n")
    paths  = images.google_images(query, jpg, png)
    print (paths)
    if(imagetype == "person"or"Person"or"PERSON"):
      size_name_background_removal.size_and_name(root_dir,query,pypath)
    else:
      size_name.size_and_name(root_dir,query,pypath)

    train_xml_and_csv_file.xml(root_dir,query)

    label_dir1 = train_tf_record.train_tf_record(root_dir,query)
    print(label_dir1)

    testcsv.testcsv(root_dir,query,test_images)

    label_dir = test_tf_record.test_tf_record(root_dir,query)
    print(label_dir)


    label_config.label(pypath,query,label_dir1,label_dir,mode,num_steps)

    if mode == "faster rncc config":
        config = "faster_rncc_inception_v2_pets.config"
        train.train(pypath,config)
    if mode == "ssd inception config":
        config = "ssd_inception_v2_coco.config"
        train.train(pypath,config)
        print("Project completed")
    if mode == "faster rncc config":
        config = "faster_rncc_inception_v2_pets.config"   
        export_inference_graph.graph(root_dir,pypath,config,num_steps,query)
        print("Project completed")
    if mode == "ssd inception config":
        config = "ssd_inception_v2_coco.config"
        export_inference_graph.graph(root_dir,pypath,config,num_steps,query)
        print("Project completed")
