import os
import json
import openslide
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


def read_property_1(simg):
    start_x = np.int(simg.properties['openslide.bounds-x'])
    start_y = np.int(simg.properties['openslide.bounds-y'])
    width_x = int(simg.properties['openslide.bounds-width'])
    height_y = int(simg.properties['openslide.bounds-height'])
    return start_x, start_y, width_x, height_y


def read_property_2(simg):
    start_x = 0
    start_y = 0
    width_x = int(simg.properties['aperio.OriginalWidth'])
    height_y = int(simg.properties['aperio.OriginalHeight'])
    return start_x, start_y, width_x, height_y


def read_property_3(simg):
    start_x = 0
    start_y = 0
    width_x = int(simg.properties['openslide.level[0].width'])
    height_y = int(simg.properties['openslide.level[0].height'])
    return start_x, start_y, width_x, height_y


def read_property_4(simg):
    start_x = 0
    start_y = 0
    width_x = int(simg.level_dimensions[0][0])
    height_y = int(simg.level_dimensions[0][1])
    return start_x, start_y, width_x, height_y


def parse_properties(simg):
    methods = [read_property_1, read_property_2, read_property_3, read_property_4]

    for method in methods:
        try:
            return method(simg)
        except Exception as e:
            print(f"Method {method.__name__} failed with error: {e}")
    raise ValueError("All methods failed to parse properties")


def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []
    for annotation in root.findall('.//Annotation'):
        for region in annotation.findall('.//Region'):
            vertices = region.find('Vertices')
            coords = [(float(vertex.get('X')), float(vertex.get('Y'))) for vertex in vertices.findall('Vertex')]
            annotations.append(coords)
    return annotations


def generate_coco_format_json(images, annotations, output_json_path):
    data = {
        "info": {
            "description": "mouse glo Dataset",
            "version": "0.1.0",
            "year": 2023,
            "date_created": "2024-01-25 16:07:08.921394"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Vanderbilt University - HRLB Lab - Dr. Yuankai Huo"
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "glomerulus",
                "supercategory": "Class"
            }
        ],
        "images": images,
        "annotations": annotations
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def crop_patches(wsi_path, annotations, output_dir, image_id_start, annotation_id_start, patch_size=2048,
                 resized_size=512, overlap=0.5, level=0):
    try:
        wsi = openslide.OpenSlide(wsi_path)
        startx, starty, width, height = parse_properties(wsi)

        try:
            downsample_factor = wsi.level_downsamples[level]
        except:
            downsample_factor = 1

        step_size = int(patch_size * (1 - overlap))

        images = []
        annotations_list = []

        annotation_id = annotation_id_start

        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        wsi_output_dir = os.path.join(output_dir, wsi_name)
        os.makedirs(wsi_output_dir, exist_ok=True)

        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                patch_annotations = []
                for coords in annotations:
                    if (x * downsample_factor <= coords[0][0] < (x + patch_size) * downsample_factor and
                        y * downsample_factor <= coords[0][1] < (y + patch_size) * downsample_factor) or (
                            x * downsample_factor <= coords[1][0] < (x + patch_size) * downsample_factor and
                            y * downsample_factor <= coords[1][1] < (y + patch_size) * downsample_factor):
                        patch_annotations.append(coords)

                if patch_annotations:
                    patch = wsi.read_region((startx + int(x * downsample_factor), starty + int(y * downsample_factor)),
                                            level,
                                            (patch_size, patch_size)).convert('RGB')
                    resized_patch = patch.resize((resized_size, resized_size), Image.ANTIALIAS)
                    image_id = image_id_start
                    file_name = os.path.join(wsi_output_dir, f"{x}_{y}.png")
                    resized_patch.save(file_name)
                    image_info = {
                        "id": image_id,
                        "file_name": file_name,
                        "width": resized_size,
                        "height": resized_size,
                        "date_captured": "2024-01-25 16:07:08.921371",
                        "license": 1,
                        "coco_url": "",
                        "flickr_url": ""
                    }
                    images.append(image_info)

                    for coords in patch_annotations:
                        left, top = coords[0]
                        right, bottom = coords[1]

                        # Scale the annotation coordinates to the resized image
                        left_resized = (left - (startx + x * downsample_factor)) * resized_size / patch_size/downsample_factor
                        top_resized = (top - (starty + y * downsample_factor)) * resized_size / patch_size/downsample_factor
                        right_resized = (right - (startx + x * downsample_factor)) * resized_size / patch_size/downsample_factor
                        bottom_resized = (bottom - (starty + y * downsample_factor)) * resized_size / patch_size/downsample_factor

                        bbox_width = right_resized - left_resized
                        bbox_height = bottom_resized - top_resized
                        bbox_area = bbox_width * bbox_height
                        center_x = (left_resized + right_resized) / 2
                        center_y = (top_resized + bottom_resized) / 2
                        radius = (bbox_width / 2 + bbox_height / 2) / 2

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "iscrowd": 0,
                            "area": bbox_area,
                            "bbox": [left_resized, top_resized, bbox_width, bbox_height],
                            "segmentation": [],
                            "width": resized_size,
                            "height": resized_size,
                            "circle_center": [center_x, center_y],
                            "circle_radius": radius
                        }
                        annotations_list.append(annotation_info)
                        annotation_id += 1

                    image_id_start += 1

        return images, annotations_list

    except Exception as e:
        print(f"Error processing WSI {os.path.basename(wsi_path)}: {e}")
        return [], []


def find_wsi_files_and_process(wsi_dir, xml_dir, output_json_path, patch_size=2048, resized_size=512, overlap=0.5,
                               level=0):
    wsi_extensions = ('.svs', '.scn', '.tiff', '.tif')

    images = []
    annotations = []

    image_id_start = 1
    annotation_id_start = 1

    output_dir = os.path.join(os.path.dirname(output_json_path), 'patches')
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(wsi_dir):
        wsi_files = [os.path.join(root, f) for f in files if f.lower().endswith(wsi_extensions)]

        for wsi_file in wsi_files:
            wsi_name = os.path.splitext(os.path.basename(wsi_file))[0].rsplit('_series', 1)[0]
            xml_file = next(
                (os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if wsi_name in f and f.lower().endswith('.xml')),
                None)

            if xml_file:
                annotations_data = parse_annotations(xml_file)
                img_data, anno_data = crop_patches(wsi_file, annotations_data, output_dir, image_id_start,
                                                   annotation_id_start, patch_size, resized_size, overlap, level)
                images.extend(img_data)
                annotations.extend(anno_data)
                image_id_start += len(img_data)
                annotation_id_start += len(anno_data)

    generate_coco_format_json(images, annotations, output_json_path)


if __name__ == "__main__":
    wsi_directory = '/data/CircleNet/data/new_data_for_miccai_paper/czi/(correct part,need rerun)28-56NX VEGF mouse study/28-56NX_tiff_series2/28-56NX_only_result(merge2)'
    xml_directory = '/data/CircleNet/data/new_data_for_miccai_paper/czi/(correct part,need rerun)28-56NX VEGF mouse study/28-56NX_tiff_series2/28-56NX_only_result(merge2)'
    output_json_path = '/data/CircleNet/data/new_data_for_miccai_paper/xml_to_json/28-56_czi_add.json'  # Set the path for the output JSON file
    find_wsi_files_and_process(wsi_directory, xml_directory, output_json_path)
