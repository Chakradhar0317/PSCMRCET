
from google_images_download import google_images_download
import os
from PIL import Image

def google_images(query, jpg, png):
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments1 = {"keywords":query,"limit":int(jpg),"format":"jpg"}
    arguments = {"keywords":query,"limit":int(png),"format":"png"} #creating list of arguments
    paths = response.download(arguments1)#passing the arguments to the function
    paths = response.download(arguments)
    return (paths)
