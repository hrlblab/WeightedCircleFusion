from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np
import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
from pathlib import Path
from lib.datasets.eval_protocals.mask import circleIOU

def merge_outputs(num_classes, max_per_image, run_nms, detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.array(detections, dtype=np.float32)

        if run_nms:
            results[j] = nms_circle(results[j], iou_th=0.35)
    scores = np.hstack(
        [results[j][:, 3] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 3] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def nms_circle(input, iou_th, merge_overlap=False):
    boxes = np.zeros((len(input), 5))
    scores = np.zeros(len(input))
    for di in range(len(input)):
        bbox_d = input[di]
        boxes[di, 0] = bbox_d[0]
        boxes[di, 1] = bbox_d[1]
        boxes[di, 2] = bbox_d[2]
        boxes[di, 3] = bbox_d[3]
        boxes[di, 4] = bbox_d[4]
        scores[di] = bbox_d[3]

    if len(scores) == 0:
       return np.hstack((boxes, scores))
    ord = np.argsort(scores)[::-1]
    scores = scores[ord]
    boxes = boxes[ord]
    sel_boxes = boxes[0][None]
    sel_scores = scores[0:1]
    for i in range(1, len(scores)):
       ious = circleIOU([boxes[i]], sel_boxes)
       if ious.max() < iou_th:
           sel_boxes = np.vstack((sel_boxes, boxes[i]))
           sel_scores = np.hstack((sel_scores, scores[i]))
       else:
           if merge_overlap:
               idx = ious.argmax()
               dim = sel_boxes.shape[1]//2
               sel_boxes[idx, :dim] = np.minimum(sel_boxes[idx, :dim], boxes[i][:dim])
               sel_boxes[idx, dim:] = np.maximum(sel_boxes[idx, dim:], boxes[i][dim:])

    # np.hstack((sel_boxes, sel_scores[:, None]))

    output = np.zeros((len(sel_boxes), 5))
    for di in range(len(output)):
        bbox_d = sel_boxes[di]
        output[di, 0] = bbox_d[0]
        output[di, 1] = bbox_d[1]
        output[di, 2] = bbox_d[2]
        output[di, 3] = bbox_d[3]
        output[di, 4] = bbox_d[4]
    return output


import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
# Assuming circleIOU and other necessary imports are correctly defined elsewhere

# Initialize detect_all with an empty array for detections
detect_all = np.empty((0, 5))

# Placeholder setup for opts, detector_factory (Assuming these are defined in your actual environment)
opt = opts().init()
Detector = detector_factory[opt.task]
detector = Detector(opt)

# Define the base path for your image dataset
base_path = Path("/data/CircleNet/data/monuseg/test")
png_files = list(base_path.glob("*.png"))
image_names = [str(file) for file in png_files]

# Placeholder for your 'info', 'licenses', 'categories' data
data1_info = {}  # Your 'info' data here
data1_licenses = []  # Your 'licenses' data here
data1_categories = []  # Your 'categories' data here

new_data = {
    'info': data1_info,
    'licenses': data1_licenses,
    'categories': data1_categories,
    'images': [],
    'annotations': []
}

image_id = 1
annotation_id = 1
num_classes = 1
scales = 1
max_per_image = 2000
run_nms = True


opt = opts().init()
Detector = detector_factory[opt.task]
detector = Detector(opt)

for (image_name) in image_names:
    ret = detector.run(image_name)
    results = ret['results']
    detect_all=[]
    #res_strs = os.path.basename(image_name).replace('.png', '').split('-x-')
    #lv_str = res_strs[0]
    #patch_start_x = np.int(res_strs[3])
    #patch_start_y = np.int(res_strs[4])

    if opt.filter_boarder:
        output_h = opt.input_h  # hard coded
        output_w = opt.input_w  # hard coded
        for j in range(1, opt.num_classes + 1):
            for i in range(len(results[j])):
                cp = [0, 0]
                cp[0] = results[j][i][0]
                cp[1] = results[j][i][1]
                cr = results[j][i][2]
                if cp[0] - cr < 0 or cp[0] + cr > output_w:
                    results[j][i][3] = 0
                    continue
                if cp[1] - cr < 0 or cp[1] + cr > output_h:
                    results[j][i][3] = 0
                    continue

    for j in range(1, opt.num_classes + 1):
        for circle in results[j]:
            #if circle[3] > opt.vis_thresh:
            if circle[3] > 0.7:
            #if circle[3] > 0:
                # circle_out = circle[:]
                # circle_out[0] = circle[0] + patch_start_x
                # circle_out[1] = circle[1] + patch_start_y
                if detect_all ==[]:
                    detect_all = [circle]
                else:
                    detect_all = np.append(detect_all, [circle], axis=0)
    results2 = merge_outputs(num_classes, max_per_image, run_nms, detect_all)
    detect_all = results2[1]

    # Add image entry to 'images' list
    new_data['images'].append({
        "id": image_id,
        "file_name": os.path.basename(image_name),
        "width": 512,
        "height": 512,
        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    })



    # Create annotations based on detect_all
    for detection in detect_all:
        centerx, centery, radius, score, category= detection
        bbox = [centerx - radius, centery - radius, 2 * radius, 2 * radius]
        area = np.pi * (radius ** 2)

        # Create and append an annotation entry
        new_data['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,  # This needs to match the correct image; you might need a mapping
            "category_id": int(category),
            "iscrowd": 0,
            "area": area,
            "bbox": bbox,  # Convert from np.array to list for JSON serialization
            "segmentation": [],  # Placeholder, fill this if you have segmentation data
            "width": 512,
            "height": 512,
            "circle_center":[centerx,centery],
            "circle_radius":radius,
            "score":score
            # Other fields as needed
        })

        annotation_id += 1
    image_id += 1



# Save to JSON file
demo_dir=opt.demo_dir
result_path=os.path.join(demo_dir,'detect_result_model1_thresh0.7.json')
with open(result_path, 'w') as outfile:
    json.dump(new_data, outfile, indent=4)



