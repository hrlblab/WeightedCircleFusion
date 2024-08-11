

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import _init_paths
import numpy as np
import os
import cv2
import sys

##for circle fusion
import subprocess
import shutil
import re

# Add the lib directory to the PYTHONPATH
src_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(src_dir, 'lib')
sys.path.append(lib_dir)

from opts import opts
from detectors.detector_factory import detector_factory
import openslide
import xmltodict
from PIL import Image
from utils.debugger import Debugger
from external.nms import soft_nms
from lib.datasets.eval_protocals.mask import circleIOU
import yaml
from easydict import EasyDict as edict

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
czi_ext = ['czi']


def get_none_zero(black_arr):

    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y

def scan_nonblack_end(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_end+offset_x, py_end), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end+offset_x, py_end), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_end, py_end+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end, py_end+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_end+(offset_x-1)
    y = py_end+(offset_y-1)
    return x,y


def get_nonblack_ending_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x - 1) * (multiples-1)
    py3 = (ending_y - 1) * (multiples-1)

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack_end(simg, px2, py2, px3, py3)

    return xx,yy

def read_property_1(simg):
    start_x = int(simg.properties['openslide.bounds-x'])
    start_y = int(simg.properties['openslide.bounds-y'])
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
    raise ValueError(f"All methods failed to parse properties: {e}")


def czi_to_patchs(args, working_dir, opt):

    patch_2d_dir = os.path.join(working_dir, '2d_patch')

    if not os.path.exists(patch_2d_dir):
        os.makedirs(patch_2d_dir)

    det_lv = args['det_lv']
    level_key = f'lv{det_lv}'
    simg = openslide.open_slide(args[level_key]['path'])




    start_x, start_y, width_x, height_y = parse_properties(simg)

    input_width = opt.input_w
    input_height = opt.input_h

    overview = simg.read_region((start_x, start_y), 0, (width_x, height_y))
    overview = np.array(overview.convert('RGB'))
    overview = cv2.cvtColor(overview, cv2.COLOR_RGB2BGR)  # opencv大坑之BGR opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！

    num_patch_x_lv = int(np.ceil(width_x/input_width))
    num_patch_y_lv = int(np.ceil(height_y/input_height))


    num_overlap_patch_x = 2 * num_patch_x_lv - 1
    num_overlap_patch_y = 2 * num_patch_y_lv - 1

    for xi in range(num_overlap_patch_x):
        for yi in range(num_overlap_patch_y):

            low_res_offset_x = int(xi * input_width / 2)
            low_res_offset_y = int(yi * input_height / 2)

            patch_start_x = start_x + int(low_res_offset_x)
            patch_start_y = start_y + int(low_res_offset_y)
            img_lv = simg.read_region((patch_start_x, patch_start_y), 0, (input_width, input_height))
            img_patch = np.array(img_lv.convert('RGB'))
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)


            # img_patch = img[patch_start_y:(patch_start_y+input_height),patch_start_x:(patch_start_x+input_width),:]
            patch_file_bname = 'lv%d-x-%04d-x-%04d-x-%d-x-%d.png' % (args['det_lv'],xi,yi,low_res_offset_x,low_res_offset_y)
            patch_file = os.path.join(patch_2d_dir, patch_file_bname)
            cv2.imwrite(patch_file, img_patch)


    return patch_2d_dir, overview, simg



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

    output = np.zeros((len(sel_boxes), 7))
    for di in range(len(output)):
        bbox_d = sel_boxes[di]
        output[di, 0] = bbox_d[0]
        output[di, 1] = bbox_d[1]
        output[di, 2] = bbox_d[2]
        output[di, 3] = bbox_d[3]
        output[di, 4] = bbox_d[4]
    return output


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


def run_one_wsi(working_dir, detector, basename, opt, xml_file, yaml_file):
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    config_dir = os.path.join(opt.config, yaml_file)

    with open(config_dir, 'r') as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    det_lv = args['det_lv']
    level_key = f'lv{det_lv}'

    # basename = os.path.basename(args.czi_file)
    #
    # basename = basename.replace('.czi', '')
    # basename = basename.replace(' ', '-')
    # working_dir = os.path.join(demo_dir, basename)
    #
    #
    #
    # xml_file = os.path.join(working_dir, '%s.xml' % (basename))
    # if os.path.exists(xml_file):
    #     return

    if os.path.exists(xml_file):
        print(f"XML file already exists: {xml_file}")
        return  # Skip processing this file

    patch_2d_dir, overview, simg = czi_to_patchs(args, working_dir, opt)

    if os.path.isdir(patch_2d_dir):
        image_names = []
        ls = os.listdir(patch_2d_dir)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(patch_2d_dir, file_name))
    else:
        image_names = [patch_2d_dir]

    detect_all = None
    count = 1
    for (image_name) in image_names:
        ret = detector.run(image_name)
        results = ret['results']
        res_strs = os.path.basename(image_name).replace('.png', '').split('-x-')
        patch_start_x = int(res_strs[3])
        patch_start_y = int(res_strs[4])

        for j in range(1, opt.num_classes + 1):
            for circle in results[j]:
                if circle[3] > opt.vis_thresh:
                    circle_out = circle[:]
                    circle_out[0] = circle[0] + patch_start_x
                    circle_out[1] = circle[1] + patch_start_y
                    if detect_all is None:
                        detect_all = [circle]
                    else:
                        detect_all = np.append(detect_all, [circle], axis=0)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(' %d/%d %s'%(count, len(image_names),time_str))
        count = count+1

    num_classes = 1
    scales = 1
    max_per_image = 2000
    run_nms = True
    results2 = merge_outputs(num_classes, max_per_image, run_nms, detect_all)
    detect_all = results2[1]


    if overview is not None:
        debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
                            theme=opt.debugger_theme)
        debugger.add_img(overview, img_id='')
        debugger.save_all_imgs(working_dir, prefix='%s' % (basename))  # save original image
        json_file = os.path.join(working_dir,'%s.json' % (basename))
        # debugger.save_detect_all_to_json(overview, detect_all, json_file, opt, simg)

        for circle in detect_all:
            debugger.add_coco_circle(circle[:3], circle[-1],
                                     circle[3], img_id='')
        debugger.save_all_imgs(working_dir, prefix='%s_overlay' % (basename))  # save original overlay




    start_x, start_y, width_x, height_y = parse_properties(simg)
    down_rate = args[level_key]['down_rate']

    detect_json = []
    doc_out = {}
    doc_out['Annotations'] = {}
    doc_out['Annotations']['@MicronsPerPixel'] = args.Pixel_width
    doc_out['Annotations']['@Level'] = level_key
    doc_out['Annotations']['@DownRate'] = down_rate
    doc_out['Annotations']['@start_x'] = start_x
    doc_out['Annotations']['@start_y'] = start_y
    doc_out['Annotations']['@width_x'] = width_x
    doc_out['Annotations']['@height_y'] = height_y
    doc_out['Annotations']['Annotation'] = {}
    doc_out['Annotations']['Annotation']['@Id'] = '1'
    doc_out['Annotations']['Annotation']['@Name'] = ''
    doc_out['Annotations']['Annotation']['@ReadOnly'] = '0'
    doc_out['Annotations']['Annotation']['@LineColorReadOnly'] = '0'
    doc_out['Annotations']['Annotation']['@Incremental'] = '0'
    doc_out['Annotations']['Annotation']['@Type'] = '4'
    doc_out['Annotations']['Annotation']['@LineColor'] = '65280'
    doc_out['Annotations']['Annotation']['@Visible'] = '1'
    doc_out['Annotations']['Annotation']['@Selected'] = '1'
    doc_out['Annotations']['Annotation']['@MarkupImagePath'] = ''
    doc_out['Annotations']['Annotation']['@MacroName'] = ''
    doc_out['Annotations']['Annotation']['Attributes'] = {}
    doc_out['Annotations']['Annotation']['Attributes']['Attribute'] = {}
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Name'] = 'glomerulus'
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Id'] = '0'
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Value'] = ''
    doc_out['Annotations']['Annotation']['Plots'] = None
    doc_out['Annotations']['Annotation']['Regions'] = {}
    doc_out['Annotations']['Annotation']['Regions']['RegionAttributeHeaders'] = {}
    doc_out['Annotations']['Annotation']['Regions']['AttributeHeader'] = []
    doc_out['Annotations']['Annotation']['Regions']['Region'] = []

    for di in range(len(detect_all)):
        detect_one = detect_all[di]
        detect_dict = {}
        detect_dict['@Id'] = str(di + 1)
        detect_dict['@Type'] = '2'
        detect_dict['@Zoom'] = '0.5'
        detect_dict['@ImageLocation'] = ''
        detect_dict['@ImageFocus'] = '-1'
        detect_dict['@Length'] = '2909.1'
        detect_dict['@Area'] = '673460.1'
        detect_dict['@LengthMicrons'] = '727.3'
        detect_dict['@AreaMicrons'] = '42091.3'
        detect_dict['@Text'] = ('%.3f' % detect_one[3])
        detect_dict['@NegativeROA'] = '0'
        detect_dict['@InputRegionId'] = '0'
        detect_dict['@Analyze'] = '0'
        detect_dict['@DisplayId'] = str(di + 1)
        detect_dict['Attributes'] = None
        detect_dict['Vertices'] = '0'
        detect_dict['Vertices'] = {}
        detect_dict['Vertices']['Vertex'] = []



        coord1 = {}
        coord1['@X'] = str((detect_one[0] - detect_one[2]))
        coord1['@Y'] = str((detect_one[1] - detect_one[2]))
        coord1['@Z'] = '0'
        coord2 = {}
        coord2['@X'] = str((detect_one[0] + detect_one[2]))  # 左右
        coord2['@Y'] = str((detect_one[1] + detect_one[2]))  # 上下
        coord2['@Z'] = '0'
        detect_dict['Vertices']['Vertex'].append(coord1)
        detect_dict['Vertices']['Vertex'].append(coord2)

        doc_out['Annotations']['Annotation']['Regions']['Region'].append(detect_dict)


    out = xmltodict.unparse(doc_out, pretty=True)
    with open(xml_file, 'wb') as file:
        file.write(out.encode('utf-8'))

    os.system('rm -r "%s"' % (os.path.join(working_dir, '2d_patch')))

    return

def test_if_fuse(demo_czi, demo_dir, opt):
    Detector = detector_factory[opt.task]


    basename = os.path.basename(demo_czi)
    basename = basename.replace('.czi', '')
    yaml_file = f"{basename}.yaml"

    working_dir = os.path.join(demo_dir, basename)

    if opt.circle_fusion:
        # Ensure load_model is treated as directory path
        model_dir = opt.load_model_dir
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            opt.load_model=model_path
            model_number_match = re.search(r'\d+', model_file)
            model_number = model_number_match.group(0) if model_number_match else ''
            xml_file = os.path.join(working_dir, f"{basename}{model_number}.xml")

            detector = Detector(opt)  # Adjust initialization as necessary

            run_one_wsi(working_dir, detector, basename, opt, xml_file, yaml_file)
            #merge result
            run_external_script('/data/CircleNet/process_code/xml_process/merge_xml.py', demo_dir=opt.demo_dir)
        copy_and_rename_files(source_directory_path=opt.demo_dir, target_directory_path=opt.target_dir)


    #evaluation for circle fusion result
    if opt.circle_fusion_eval:
        run_external_script('/data/CircleNet/process_code/xml_process/eval_xml.py', demo_dir=opt.demo_dir, gt_dir=opt.gt_dir)


            # Your prediction and XML saving logic here

    else:
        # Your existing single model logic here
        model_filename = os.path.basename(opt.load_model)
        model_number_match = re.search(r'\d+', model_filename)
        model_number = model_number_match.group(0) if model_number_match else ''
        xml_file = os.path.join(working_dir, f"{basename}{model_number}.xml")

        detector = Detector(opt)  # Assuming detector initialization works as before

        run_one_wsi(working_dir, detector, basename, opt, xml_file)


        # Your prediction and XML saving logic here

##Executes an external Python script for circle fusion with optional arguments
# def run_external_script(script_path, **kwargs):
#     """
#     Executes an external Python script with optional named arguments.
#
#     :param script_path: Path to the Python script to be executed.
#     :param kwargs: Dictionary of named arguments to pass to the script.
#     """
#     try:
#         # Convert named arguments into command line arguments
#         args = []
#         for key, value in kwargs.items():
#             args.append(f"--{key}")
#             args.append(value)
#
#         command = ["python", script_path] + args
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while executing {script_path}: {e}")

def run_external_script(script_path, **kwargs):
    """
    Executes an external Python script with optional named arguments.

    :param script_path: Path to the Python script to be executed.
    :param kwargs: Dictionary of named arguments to pass to the script.
    """
    try:
        # Convert named arguments into command line arguments
        args = []
        for key, value in kwargs.items():
            args.append(f"--{key}")
            args.append(value)

        command = ["python", script_path] + args
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print standard output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_path}: {e}")
        print(e.stderr)  # Print standard error output

####### move the final result to another folder ###################################
def copy_and_rename_files(source_directory_path, target_directory_path):
    for root, dirs, files in os.walk(source_directory_path):
        for file in files:
            if file.endswith('.xml') and 'mergecount2_iou' in file:
                source_file_path = os.path.join(root, file)

                # Remove "mergecount2_iou" from the file name
                new_file_name = file.replace('mergecount2_iou', '')
                target_file_path = os.path.join(target_directory_path, new_file_name)

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                shutil.copy(source_file_path, target_file_path)
                print(f'Copied: {source_file_path} to {target_file_path}')


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    demo_dir = opt.demo_dir


    if os.path.isdir(opt.demo):
        czi_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in czi_ext:
                czi_names.append(os.path.join(opt.demo, file_name))
    else:
        czi_names = [opt.demo]

    for (demo_czi) in czi_names:
        try:
            #opt.lv=1
            #run_one_scn(demo_scn, demo_dir, opt)
            opt.lv=2
            test_if_fuse(demo_czi, demo_dir, opt)
        except Exception as e:
            print(f"An error occurred while running scenario '{demo_czi}': {e}")

    #run_one_wsi(demo_dir, opt)






if __name__ == '__main__':

    opt = opts().init()
    demo(opt)


