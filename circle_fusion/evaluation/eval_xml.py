import xml.etree.ElementTree as ET
import math
import os

import argparse

def parse_xml(file_path):
    """
    Parse an XML file and extract region vertices.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    regions = []
    for region in root.findall('.//Region'):
        vertices = region.find('Vertices')
        vertex_list = vertices.findall('Vertex')
        if len(vertex_list) >= 2:  # Ensure there are at least two vertices to define a region
            x1, y1 = float(vertex_list[0].get('X')), float(vertex_list[0].get('Y'))
            x2, y2 = float(vertex_list[1].get('X')), float(vertex_list[1].get('Y'))
            text = region.get('Text')

            regions.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return regions


def rectangle_to_circle(x1, y1, x2, y2):
    # Calculate the center of the circle
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    # Calculate the radius of the circle
    radius = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
    return center_x, center_y, radius


def circle_area_intersection(circleA, circleB):
    x0, y0, r0 = circleA
    x1, y1, r1 = circleB
    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # No overlap
    if d >= (r0 + r1):
        return 0
    # One circle within another
    elif d <= abs(r0 - r1) and r0 < r1:
        return math.pi * r0 ** 2
    elif d <= abs(r0 - r1) and r0 >= r1:
        return math.pi * r1 ** 2
    else:
        # Calculate the area of overlap between the two circles
        phi = (math.acos((r0 ** 2 + d ** 2 - r1 ** 2) / (2 * r0 * d))) * 2
        theta = (math.acos((r1 ** 2 + d ** 2 - r0 ** 2) / (2 * r1 * d))) * 2
        area1 = 0.5 * theta * r1 ** 2 - 0.5 * r1 ** 2 * math.sin(theta)
        area2 = 0.5 * phi * r0 ** 2 - 0.5 * r0 ** 2 * math.sin(phi)
        return area1 + area2


def calculate_iou(boxA, boxB):
    # Convert boxes to circles
    circleA = rectangle_to_circle(**boxA)
    circleB = rectangle_to_circle(**boxB)

    # Calculate the area of both circles
    areaA = math.pi * (circleA[2] ** 2)
    areaB = math.pi * (circleB[2] ** 2)

    # Calculate the area of intersection
    interArea = circle_area_intersection(circleA, circleB)

    # Calculate the total area covered by both circles
    totalArea = areaA + areaB - interArea

    # Calculate the IoU
    iou = interArea / totalArea
    return iou



def calculate_average_precision_and_recall(detections, ground_truths, iou_thresholds):
    average_precisions = []
    recalls = []

    for threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = len(ground_truths)  # Initially, all ground truths are considered not detected

        matched_ground_truths = set()

        for detection in detections:
            matched = False
            for gt_index, gt in enumerate(ground_truths):
                if gt_index in matched_ground_truths:
                    continue  # Skip ground truths that have already been matched

                iou = calculate_iou(detection, gt)
                if iou >= threshold:
                    matched = True
                    tp += 1
                    matched_ground_truths.add(gt_index)
                    break  # Stop searching once a match is found for this detection

            if not matched:
                fp += 1

        # Now, fn is simply the ground truths that were never matched
        fn = len(ground_truths) - len(matched_ground_truths)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = (len(ground_truths) -fn) / len(ground_truths) if len(ground_truths) > 0 else 0

        average_precisions.append(precision)
        recalls.append(recall)

    map = sum(average_precisions) / len(average_precisions)
    average_recall = sum(recalls) / len(recalls)

    return map, average_recall

# def find_corresponding_file(gt_file, detection_folder):
#     # Assuming gt_file format is "204550.xml" and detection format is "204550_merge.xml"
#     base_name = os.path.splitext(gt_file)[0]  # e.g., 204550 from 204550.xml
#     first_part = base_name.split("_")[0]
#     detection_file = f"{first_part}mergecount2_iou.xml"
#     detection_path = os.path.join(detection_folder, detection_file)
#     if os.path.exists(detection_path):
#         return detection_path
#     else:
#         return None

def find_corresponding_file(gt_file, detection_folder):
    # Assuming gt_file format is "204550.xml" and detection format is in a subfolder named "204550" as "204550_mergecount2_iou.xml"
    base_name = os.path.splitext(gt_file)[0]  # e.g., 204550 from 204550.xml
    first_part = base_name.split("_")[0]  # Assuming the format "204550" from "204550.xml"
    # Adjusted to include the subfolder in the path
    subfolder_path = os.path.join(detection_folder, first_part)  # Path to the subfolder
    #detection_file = f"{first_part}mergecount2_iou.xml"
    # Assuming this is the correct file naming convention
    detection_file = (f"{first_part}mergecount(96581)_iou.xml")
    detection_path = os.path.join(subfolder_path, detection_file)  # Full path to the detection file

    if os.path.exists(detection_path):
        return detection_path
    else:
        return None


def calculate_ap_for_folders(gt_folder, detection_folder):
    iou_thresholds = [0.5, 0.75]
    mean_iou_thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    sum_map05=0
    sum_map075=0
    sum_mapmean=0
    sum_recall=0
    filenum=0

    for gt_file in os.listdir(gt_folder):
        if gt_file.endswith(".xml"):
            gt_path = os.path.join(gt_folder, gt_file)
            detection_path = find_corresponding_file(gt_file, detection_folder)
            if detection_path:
                gt_data = parse_xml(gt_path)
                detection_data = parse_xml(detection_path)

                # Calculate AP at IoU 0.5 and 0.75 for this pair
                for iou_threshold in iou_thresholds:
                    map, average_recall = calculate_average_precision_and_recall(detection_data, gt_data, [iou_threshold])
                    print(f"File: {gt_file}, IoU: {iou_threshold}, AP: {map}")
                    if iou_threshold==0.5:
                        sum_map05+=map
                    else:
                        sum_map075+=map


                map_mean, average_recall_mean = calculate_average_precision_and_recall(detection_data, gt_data, mean_iou_thresholds)
                print(f"File: {gt_file}, IOU:0.5:0.95, AP: {map_mean}")
                print(f"File: {gt_file}, IoU:0.5:0.95, AR: {average_recall_mean}")

                sum_mapmean+=map_mean
                sum_recall+=average_recall_mean
                filenum+=1

    mAP_allsvs=sum_mapmean/filenum
    mAR_allsvs=sum_recall/filenum
    map_05=sum_map05/filenum
    map_075=sum_map075/filenum
    print(f"all xml: IOU=0.5:0.95,mAP:{mAP_allsvs}")
    print(f"all xml: IOU=0.5:0.95,mAR:{mAR_allsvs}")
    print(f"all xml: IOU=0.5,mAP:{map_05}")
    print(f"all xml: IOU=0.75,mAP:{map_075}")


parser = argparse.ArgumentParser(description="Second script processing")
parser.add_argument("--demo_dir", required=True, help="Directory for demo")
parser.add_argument("--gt_dir", required=True, help="Ground truth directory")
args = parser.parse_args()

# Example usage
gt_folder = args.gt_dir
detection_folder = args.demo_dir
calculate_ap_for_folders(gt_folder, detection_folder)
