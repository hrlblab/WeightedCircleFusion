import xml.etree.ElementTree as ET
import math
import os
import re


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

            regions.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'count': 0 , 'text': text})
    return regions


def calculate_circle(region):
    """
    Calculate the center and radius of a circle from the bounding vertices.
    """
    x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    radius = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
    return center_x, center_y, radius



# def split_circle(center_x, center_y, radius):
#     """
#     Split a circle into two smaller circles with half of its radius and centers shifted to the left and right.
#     """
#     # Left half circle
#     left_x1 = center_x - radius / 2
#     left_y1 = center_y + radius / 2
#     left_x2 = center_x
#     left_y2 = center_y - radius / 2
#
#     # Right half circle
#     right_x1 = center_x
#     right_y1 = center_y + radius / 2
#     right_x2 = center_x + radius / 2
#     right_y2 = center_y - radius / 2
#
#     return {'x1': left_x1, 'y1': left_y1, 'x2': left_x2, 'y2': left_y2}, {'x1': right_x1, 'y1': right_y1, 'x2': right_x2, 'y2': right_y2}

def circle_iou(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # No overlap
    if d >= r1 + r2:
        return 0
    # One circle is completely within the other
    if d <= abs(r1 - r2) and r1 < r2:
        overlap_area = math.pi * r1 ** 2
    elif d <= abs(r1 - r2) and r2 < r1:
        overlap_area = math.pi * r2 ** 2
    else:
        # Partial overlap
        part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        overlap_area = part1 + part2 - part3

    # Area of each circle
    area1 = math.pi * r1 ** 2
    area2 = math.pi * r2 ** 2

    # Total area covered by both circles
    total_area = area1 + area2 - overlap_area

    # Intersection over Union (IoU)
    iou = overlap_area / total_area

    return iou


def weighted_average(value1, weight1, value2, weight2):
    return (value1 * weight1 + value2 * weight2) / (weight1 + weight2)

def merge_regions(existing_region, new_region):
    # Convert 'text' to numeric values for weighting
    existing_text_weight = float(existing_region['text'])
    new_text_weight = float(new_region['text'])

    # Calculate weighted averages for coordinates
    existing_region['x1'] = weighted_average(existing_region['x1'], existing_text_weight, new_region['x1'], new_text_weight)
    existing_region['y1'] = weighted_average(existing_region['y1'], existing_text_weight, new_region['y1'], new_text_weight)
    existing_region['x2'] = weighted_average(existing_region['x2'], existing_text_weight, new_region['x2'], new_text_weight)
    existing_region['y2'] = weighted_average(existing_region['y2'], existing_text_weight, new_region['y2'], new_text_weight)

    # Update 'text' based on average of existing and new 'text' values
    total_weight = existing_text_weight + new_text_weight
    existing_region['text'] = str(round(total_weight / 2, 2))

# def weighted_average(value1, weight1, value2, weight2):
#     return (value1 * weight1 + value2 * weight2) / (weight1 + weight2)
#
# def merge_regions(existing_region, new_region):
#     # Convert 'text' to numeric values for weighting
#     existing_text_weight = float(existing_region['text'])
#     new_text_weight = float(new_region['text'])
#
#     # Calculate weighted averages for circle center coordinates
#     existing_region['center_x'] = weighted_average(existing_region['center_x'], existing_text_weight, new_region['center_x'], new_text_weight)
#     existing_region['center_y'] = weighted_average(existing_region['center_y'], existing_text_weight, new_region['center_y'], new_text_weight)
#
#     # Calculate weighted average for radius
#     existing_region['radius'] = weighted_average(existing_region['radius'], existing_text_weight, new_region['radius'], new_text_weight)
#
#     # Update 'text' based on the total weight
#     total_weight = existing_text_weight + new_text_weight
#     existing_region['text'] = str(round(total_weight, 2)) # You may want to keep this as a float or integer depending on your use case

def is_circle_within_another(center1, radius1, center2, radius2):
    """Check if one circle is completely within another."""
    distance_centers = math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)
    # Determine if one circle is within another, considering their radii and distance between centers.
    return distance_centers + min(radius1, radius2) <= max(radius1, radius2), radius1 <= radius2


# def filter_overlapping_circles(existing_regions):
#     """Filter out larger circles that completely encompass smaller ones from existing_regions."""
#     # Create a list of tuples containing circle center and radius for easier comparison.
#     circles = [(calculate_circle(region), i) for i, region in enumerate(existing_regions)]
#     to_remove = set()
#
#     for i, (circle1, idx1) in enumerate(circles):
#         for circle2, idx2 in circles[i+1:]:
#             within, is_first_smaller_or_equal = is_circle_within_another(circle1[:2], circle1[2], circle2[:2], circle2[2])
#             if within:
#                 # Add the index of the larger circle to the removal set.
#                 larger_circle_idx = idx1 if not is_first_smaller_or_equal else idx2
#                 to_remove.add(larger_circle_idx)
#
#     # Generate a new list excluding the indices marked for removal.
#     filtered_regions = [region for i, region in enumerate(existing_regions) if i not in to_remove]
#     return filtered_regions

def filter_overlapping_circles(existing_regions):
    """Filter out larger circles that completely encompass smaller ones from existing_regions,
    and also filter out elements where 'text' < 0.4 and 'count' < 0.3."""
    circles = [(calculate_circle(region), i) for i, region in enumerate(existing_regions)]
    to_remove = set()

    for i, (circle1, idx1) in enumerate(circles):
        # Check if the 'text' and 'count' values of the current circle meet the removal criteria
        if (float(existing_regions[idx1]['text']) < 0.9 and float(existing_regions[idx1]['count']) < 2)\
                or (float(existing_regions[idx1]['text']) < 0.1 and 1<float(existing_regions[idx1]['count']) < 3)\
                or (float(existing_regions[idx1]['text']) < 0.1 and 2<float(existing_regions[idx1]['count']) < 4)\
                or (float(existing_regions[idx1]['text']) < 0.1 and 3<float(existing_regions[idx1]['count']) < 5):
            to_remove.add(idx1)
            continue  # Skip further checks and proceed to the next iteration

        for circle2, idx2 in circles[i + 1:]:
            within, is_first_smaller_or_equal = is_circle_within_another(circle1[:2], circle1[2], circle2[:2],
                                                                         circle2[2])
            if within:
                # Determine the larger circle based on whether the first is smaller or equal
                larger_circle_idx = idx1 if not is_first_smaller_or_equal else idx2

                # Ensure we do not remove circles based on size if they should be removed based on 'text' and 'count'
                if not (
                        (float(existing_regions[larger_circle_idx]['text']) < 0.9 and float(
                            existing_regions[larger_circle_idx]['count']) < 2) or
                        (float(existing_regions[larger_circle_idx]['text']) < 0.1 and 1< float(
                            existing_regions[larger_circle_idx]['count']) < 3) or
                        (float(existing_regions[larger_circle_idx]['text']) < 0.1 and 2 < float(
                            existing_regions[larger_circle_idx]['count']) < 4)or
                        (float(existing_regions[larger_circle_idx]['text']) < 0.1 and 3 < float(
                            existing_regions[larger_circle_idx]['count']) < 5)
                ):
                    to_remove.add(larger_circle_idx)

    # Generate a new list excluding the indices marked for removal.
    filtered_regions = [region for i, region in enumerate(existing_regions) if i not in to_remove]
    return filtered_regions
def create_xml_from_regions(regions, file_name, microns_per_pixel, level, down_rate, start_x, start_y, width_x, height_y, device):
    """
    Create an XML file from the list of region dictionaries,
    including specific structure at the start and end of the file.
    """
    # Extract unique count values
    unique_counts = set(region['count'] for region in regions)

    # Create the root element
    annotations = ET.Element('Annotations',
                             MicronsPerPixel=str(microns_per_pixel),
                             Level=str(level),
                             DownRate=str(down_rate),
                             start_x=str(start_x),
                             start_y=str(start_y),
                             width_x=str(width_x),
                             height_y=str(height_y),
                             Device=device)

    # Define line colors based on count
    line_colors = {0: "255", 1: "16711680", 2: "16776960", 3: "65280", 4: "32768"}

    # Define text values based on count
    text_values = {0: '0.2', 1: '0.4', 2: '0.6', 3: '0.8', 4: '1.0',}

    # Create annotations for each unique count value
    for count in unique_counts:
        # Filter regions with the current count
        count_regions = [region for region in regions if region['count'] == count]

        # Create the Annotation element
        annotation = ET.SubElement(annotations, 'Annotation',
                                   Id=str(count), Name="", ReadOnly="0",
                                   LineColorReadOnly="0", Incremental="0",
                                   Type="4", LineColor=line_colors.get(count, "255"), Visible="1",
                                   Selected="1", MarkupImagePath="", MacroName="")

        # Attributes (could be dynamic based on your requirements)
        attributes = ET.SubElement(annotation, 'Attributes')
        ET.SubElement(attributes, 'Attribute', Name="glomerulus", Id="0", Value="")

        # Plots (empty as per your structure)
        plots = ET.SubElement(annotation, 'Plots')

        # Regions
        regions_element = ET.SubElement(annotation, 'Regions')
        region_attribute_headers = ET.SubElement(regions_element, 'RegionAttributeHeaders')

        for idx, region in enumerate(count_regions, start=1):
            text = text_values.get(count, '0.2')
            region_elem = ET.SubElement(regions_element, 'Region', Id=str(idx), Type="2", Zoom="0.5", ImageLocation="",
                                        ImageFocus="-1", Length="2909.1", Area="673460.1", LengthMicrons="727.3",
                                        AreaMicrons="42091.3", Text=str(text)+'||'+region['text'], NegativeROA="0",
                                        InputRegionId="0", Analyze="0", DisplayId=str(idx))
            vertices_elem = ET.SubElement(region_elem, 'Vertices')
            ET.SubElement(vertices_elem, 'Vertex', X=str(region['x1']), Y=str(region['y1']), Z="0")
            ET.SubElement(vertices_elem, 'Vertex', X=str(region['x2']), Y=str(region['y2']), Z="0")

    # Convert the ElementTree to a string and then save it to a file
    tree = ET.ElementTree(annotations)
    tree.write(file_name, encoding='utf-8', xml_declaration=True)





def process_xml_files(xml_files):
    """
    Process each XML file, filter and append regions, and generate a final XML.
    """
    existing_regions = []  # Initialize with regions from the first file

    # Extract information from the first XML file
    if xml_files:
        with open(xml_files[0], 'r') as first_xml:
            first_tree = ET.parse(first_xml)
            first_root = first_tree.getroot()
            microns_per_pixel = float(first_root.attrib['MicronsPerPixel'])
            # Extract the Level attribute
            #level = int(first_root.attrib['Level'])
            down_rate = float(first_root.attrib['DownRate'])
            start_x = int(first_root.attrib['start_x'])
            start_y = int(first_root.attrib['start_y'])
            width_x = int(first_root.attrib['width_x'])
            height_y = int(first_root.attrib['height_y'])
            #device = first_root.attrib['Device']

            # Extract the Level attribute
            level_str = first_root.attrib['Level']

            # Extract only the numeric part of the string
            level_num_str = re.search(r'\d+', level_str).group()

            # Convert the numeric part to an integer
            level = int(level_num_str)





    for file_index, xml_file in enumerate(xml_files):
        new_regions = parse_xml(xml_file)
        if file_index > 0:  # For subsequent files, apply the filtering logic
            filtered_regions = []
            for new_region in new_regions:
                center_x, center_y, radius = calculate_circle(new_region)
                overlap_found = False
                area = math.pi * radius ** 2
                # if area > 1200000:
                #     left_region, right_region = split_circle(center_x, center_y, radius)
                #     new_regions.append(left_region)
                #     new_regions.append(right_region)
                #     new_regions.remove(new_region)
                #     continue


                for existing_region in existing_regions:
                    try:
                        ex_center_x, ex_center_y, ex_radius = calculate_circle(existing_region)
                        IOU = circle_iou((center_x, center_y, radius),
                                                           (ex_center_x, ex_center_y, ex_radius))
                    except Exception as e:
                        print(f"Error calculating overlap area for region: {existing_region}. Skipping. Error: {e}")
                        continue

                    if IOU > 0.5:
                        if existing_region['count']<4:
                            existing_region['count']+=1
                        else:
                            existing_region['count']=0

                        overlap_found = True
                        # existing_circle = {'center_x': ex_center_x, 'center_y': ex_center_y, 'radius': ex_radius, 'text': existing_region['text']}
                        # new_circle = {'center_x': center_x, 'center_y': center_y, 'radius': radius, 'text': new_region['text']}
                        # merge_regions(existing_circle,new_circle)
                        merge_regions(existing_region, new_region)
                        #existing_region['text'] = str(round((float(existing_region['text']) + float(new_region['text'])) / 2, 2))


                        break

                if not overlap_found:
                    filtered_regions.append(new_region)
            existing_regions.extend(filtered_regions)
        else:
            existing_regions.extend(new_regions)  # For the first file, just add regions

    filtered_exist_regions = filter_overlapping_circles(existing_regions)


    # Generate the output file name using the folder name
    folder_path = os.path.dirname(xml_files[0])
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(folder_path, folder_name + "mergecount(9111)_iou.xml")

    # Generate and save the final XML
    create_xml_from_regions(filtered_exist_regions, output_file, microns_per_pixel, level, down_rate, start_x, start_y, width_x, height_y, device="unknown")


def process_multiple_folders(root_folder):
    """
    Process multiple folders each containing multiple XML files.
    """
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            xml_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                         if f.endswith('.xml') and "merge" not in f.lower() and "nms" not in f.lower()]

            if xml_files:
                process_xml_files(xml_files)

# # List of XML file paths
# xml_files = [
#     '/data/CircleNet/data/five_fold_json/show_on_svs/mouse_glo/test_merge_xml/WAX-G1-6_1.xml',
#     '/data/CircleNet/data/five_fold_json/show_on_svs/mouse_glo/test_merge_xml/WAX-G1-6_2.xml',
#     '/data/CircleNet/data/five_fold_json/show_on_svs/mouse_glo/test_merge_xml/WAX-G1-6_3.xml',
#     '/data/CircleNet/data/five_fold_json/show_on_svs/mouse_glo/test_merge_xml/WAX-G1-6_4.xml',
#     '/data/CircleNet/data/five_fold_json/show_on_svs/mouse_glo/test_merge_xml/WAX-G1-6_5.xml'
# ]

parser = argparse.ArgumentParser(description="External Script Description")
parser.add_argument("--demo_dir", required=True, help="Directory for demo")

args = parser.parse_args()

root_folder = args.demo_dir


# Process the XML files
process_multiple_folders(root_folder)

