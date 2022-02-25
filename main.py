import numpy as np
import os
import sys
import mlflow
import json
import torch
from PIL import Image
import glob


from git_utils import get_git_revision_short_hash, get_git_url

# get the git hash of the current commit
short_hash = get_git_revision_short_hash()
git_url = get_git_url()

import argparse

def predict(images, yolo_type='yolov5s'):
    full_result = []

    # Model
    model = torch.hub.load('ultralytics/yolov5', yolo_type)  # or yolov5m, yolov5l, yolov5x, custom

    # Inference
    results = model(images)

    for image_detections in results.xyxy:
        image_segmentation = []
        # extract the detections for every image
        for det in image_detections:
            x1,y1,x2,y2,conf,label = det.cpu().detach().numpy()
            x1,y1,x2,y2 = map(lambda x: int(np.round(x).astype(np.int32)), [x1,y1,x2,y2])

            image_segmentation.append(
                dict(
                    label = f'{int(label)}',
                    contour_coordinates = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                    type = 'Polygon'
                )
            )

        # append to result list
        full_result.append(image_segmentation)

    return full_result


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('images', type=str, nargs='+',
                    help='list of images')
parser.add_argument('--yolo-type', default="yolov5s", help="Use the omnipose model")

args = parser.parse_args()

if len(args.images) == 1:
    image_path = args.images[0]
    if os.path.isdir(image_path):
        # it's a folder, iterate all images in the folder
        args.images = sorted(glob.glob(os.path.join(image_path, '*.png')))
    else:
        # it may be a list of images
        args.images = image_path.split(' ')

images = [np.asarray(Image.open(image_path)) for image_path in args.images]

yolo_type = args.yolo_type

result = predict(images, yolo_type)

if len(images) == 1:
    result = result[0]

result = dict(
    model = f'{git_url}@{short_hash}',
    format_version = '0.2', # version of the segmentation format
    segmentation_data = result # [[Detection,...]]
)

with open('output.json', 'w') as output:
    json.dump(result, output)

mlflow.log_artifact('output.json')
