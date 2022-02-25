import numpy as np
import os
import sys
import mlflow
import json
import torch
from PIL import Image
import glob


from git_utils import CACHE_FOLDER, cached_file, get_git_revision_short_hash, get_git_url

# get the git hash of the current commit
short_hash = get_git_revision_short_hash()
git_url = get_git_url()

import argparse

COCO_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

def predict(images, yolo_type='yolov5s', model_path=None, labels=COCO_LABELS):
    full_result = []

    if model_path:
        # TODO: caching
        checkpoint_path = cached_file(model_path, cache_folder=CACHE_FOLDER)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_path, source='local')
    else:
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
                    label = f'{labels[int(label)]}',
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
parser.add_argument('--model-path', help="custom path/url of a model")

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

result = predict(images, yolo_type, args.model_path)

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
