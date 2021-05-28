import re
import os
import cv2
import json
import itertools
import numpy as np
import glob
import scipy.io as sio
from pycocotools import mask as cocomask
from PIL import Image
import matplotlib.pyplot as plt

categories = []
items = ["supercategory",
        "name",
        "id"]

def makeCategories(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      #print(l)
      name = l.strip()
      temp = {items[0]:"none",items[1]:name,items[2]:i}
      categories.append(temp)
  return categories

categories = makeCategories('/content/detr_tutorial/dataset/data/classes.txt')

#categories = [{"supercategory": "none","name": "face","id": 0}]

phases = ["train","valid","test"]
for phase in phases:
    root_path = "data/{}/".format(phase)
    #gt_path = os.path.join("wider_face_split/wider_face_{}.mat".format(phase))
    json_file = "data/{}.json".format(phase)

    #gt = sio.loadmat(gt_path)
    #event_list = gt.get("event_list")
    #file_list = gt.get("file_list")
    #face_bbox_list = gt.get("face_bbx_list")

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annot_count = 0
    image_id = 0
    processed = 0

    #Obtain label and img name
    file_list = glob.glob(root_path+"*.txt")
    img_paths = []
    filenames = []
    for i,val in enumerate(file_list):
        img_paths.append(file_list[i].replace(".txt",".jpg"))
        filenames.append(file_list[i].replace(".txt",".jpg").replace(root_path,""))
        #Image annotation
        img = cv2.imread(img_paths[i])
        img_h, img_w, channels = img.shape
        img_elem = {"file_name": filenames[i],
                    "height": img_h,
                    "width": img_w,
                    "id": image_id}

        res_file["images"].append(img_elem)
        with open(val,"r") as f:
            for line in f.readlines():
                key = line.strip()
                coords = key.split()

                x_center = (float(coords[1])*(img_w))
                y_center = (float(coords[2])*(img_h))
                width = (float(coords[3])*img_w)
                height = (float(coords[4])*img_h)
                category_id = int(coords[0]) # label

                mid_x = int(x_center-width/2)
                mid_y = int(y_center-height/2)
                width = int(width)
                height = int(height)

                area = width*height
                poly = [[mid_x, mid_y],
                        [width, height],
                        [width, height],
                        [mid_x, mid_y]]

                annot_elem = {
                    "id": annot_count,
                    "bbox": [
                        float(mid_x),
                        float(mid_y),
                        float(width),
                        float(height)
                    ],
                    "segmentation": list([poly]),
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": float(area)
                }
                res_file["annotations"].append(annot_elem)
                annot_count += 1

            image_id += 1

            processed += 1

    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")
