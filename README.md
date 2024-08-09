# WeightedCircleFusion
To run it on Whole Slide Images, please go to `circle_fusion` folder and run:
'''
python /data/CircleNet/src/run_detection_for_scn.py circledet --circle_fusion --circle_fusion_eval --demo "/data/CircleNet/circle_fusion_test/WSI" --load_model_dir "/data/CircleNet/exp/circledet/cdf_models_new" --filter_boarder --demo_dir "/data/CircleNet/circle_fusion_test/result" --gt_dir "/data/CircleNet/circle_fusion_test/ground_truth" 
'''
demo is the path to the WSI images that you want to process. load_model_dir is the path to your models' folder. demo_dir is where your result finally generate. gt_dir is the path to your ground truths,which are xml files here.
You can set them anywhere in your computer.
