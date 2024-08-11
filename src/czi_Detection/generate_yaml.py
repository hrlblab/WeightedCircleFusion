# import os
# import yaml
#
# # Define the directories for CZI and TIFF files
# czi_dir = "/data/CircleNet/data/new_data_for_miccai_paper/czi/20-DN mouse model-LJ Ma"
# tiff_dir_lv0 = "/data/CircleNet/data/new_data_for_miccai_paper/czi/20-DN mouse model-LJ Ma/20-DN_tiff_series2"
# tiff_dir_lv2 = "/data/CircleNet/data/new_data_for_miccai_paper/czi/20-DN mouse model-LJ Ma/20-DN_tiff_series2"
#
# # Define the directory to save YAML files
# yaml_dir = "/data/CircleNet/data/new_data_for_miccai_paper/czi/20-DN mouse model-LJ Ma/20-DN_yaml"
#
# # Ensure the YAML directory exists
# os.makedirs(yaml_dir, exist_ok=True)
#
# # Define the pixel dimensions and down rates
# pixel_width = 0.1109
# pixel_height = 0.1109
# down_rates = {0: 0, 2: 4, 3: 8, 4: 16}
#
#
# # Create a function to generate the YAML content for a given file
# def generate_yaml(czi_file):
#     base_name = os.path.splitext(os.path.basename(czi_file))[0]
#
#     yaml_content = {
#         "czi_file": czi_file,
#         "det_lv": 2,
#         "lv0": {
#             "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
#             "down_rate": down_rates[0]
#         },
#         "lv2": {
#             "path": os.path.join(tiff_dir_lv2, f"{base_name}_series2.tiff"),
#             "down_rate": down_rates[2]
#         },
#         "lv3": {
#             "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
#             "down_rate": down_rates[3]
#         },
#         "lv4": {
#             "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
#             "down_rate": down_rates[4]
#         },
#         "Pixel_width": pixel_width,
#         "Pixel_height": pixel_height
#     }
#
#     return yaml_content
#
#
# # Loop through all CZI files and generate corresponding YAML files
# for czi_file in os.listdir(czi_dir):
#     if czi_file.endswith(".czi"):
#         czi_path = os.path.join(czi_dir, czi_file)
#         yaml_content = generate_yaml(czi_path)
#
#         # Write the YAML content to a file
#         yaml_file = os.path.join(yaml_dir, f"{os.path.splitext(czi_file)[0]}.yaml")
#         with open(yaml_file, 'w') as f:
#             yaml.dump(yaml_content, f, default_flow_style=False)
#
#         print(f"Generated YAML for {czi_file}")
#
# print("All YAML files generated.")

import os
import yaml

# Define the directories for CZI and TIFF files
czi_dir = "/data/CircleNet/data/new_data_for_miccai_paper/czi/31-Gilead 56NX study"
tiff_dir_lv0 = "/data/CircleNet/data/new_data_for_miccai_paper/czi/31-Gilead 56NX study/31-Gilead_tiff_series2"
tiff_dir_lv2 = "/data/CircleNet/data/new_data_for_miccai_paper/czi/31-Gilead 56NX study/31-Gilead_tiff_series2"

# Define the directory to save YAML files
yaml_dir = "/data/CircleNet/data/new_data_for_miccai_paper/czi/31-Gilead 56NX study/31-Gilead_yaml"

# Ensure the YAML directory exists
os.makedirs(yaml_dir, exist_ok=True)

# Define the pixel dimensions and down rates
pixel_width = 0.1109
pixel_height = 0.1109
down_rates = {0: 0, 2: 4, 3: 8, 4: 16}

# Create a function to generate the YAML content for a given file
def generate_yaml(czi_file):
    base_name = os.path.splitext(os.path.basename(czi_file))[0]

    yaml_content = {
        "czi_file": czi_file,
        "det_lv": 2,
        "lv0": {
            "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
            "down_rate": down_rates[0]
        },
        "lv2": {
            "path": os.path.join(tiff_dir_lv2, f"{base_name}_series2.tiff"),
            "down_rate": down_rates[2]
        },
        "lv3": {
            "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
            "down_rate": down_rates[3]
        },
        "lv4": {
            "path": os.path.join(tiff_dir_lv0, f"{base_name}_series2.tiff"),
            "down_rate": down_rates[4]
        },
        "Pixel_width": pixel_width,
        "Pixel_height": pixel_height
    }

    return yaml_content

# Loop through all CZI files and generate corresponding YAML files
for czi_file in os.listdir(czi_dir):
    if czi_file.endswith(".czi") and "_pt" not in czi_file:
        czi_path = os.path.join(czi_dir, czi_file)
        yaml_content = generate_yaml(czi_path)

        # Write the YAML content to a file
        yaml_file = os.path.join(yaml_dir, f"{os.path.splitext(czi_file)[0]}.yaml")
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"Generated YAML for {czi_file}")

print("All YAML files generated.")


