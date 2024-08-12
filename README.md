# WeightedCircleFusion
## Abstract
![](docs/problem1.pdf)
Recently, the use of circle representation has emerged as a method to improve the identification of spherical objects (such as glomeruli, cells, and nuclei) in medical imaging studies. In traditional bounding box-based object detection, combining results from multiple models improves accuracy, especially when real-time processing isn’t crucial. Unfortunately, this widely adopted strategy is not readily available for combining circle representations. In this paper, we propose Weighted Circle Fusion (WCF), a simple approach for merging predictions from various circle detection models. Our method leverages confidence scores associated with each proposed bounding circle to generate averaged circles. We evaluate our method on a proprietary dataset for glomerular detection in whole slide imaging (WSI) and find a performance gain of 5% compared to existing ensemble methods. Additionally, we assess the efficiency of two annotation methods—fully manual annotation and a human-in-the-loop (HITL) approach—in labeling 200,000 glomeruli. The HITL approach, which integrates machine learning detection with human verification, demonstrated remarkable improvements in annotation efficiency. The Weighted Circle Fusion technique not only enhances object detection precision but also notably reduces false detections, presenting a promising direction for future research and application in pathological image analysis.
## Model Training
This experiment uses CircleNet to train new models. Please refer to the [CircleNet](https://github.com/hrlblab/CircleNet/blob/master/README.md) for instructions on how to train new models.

This is a collection of models trained from my previous circle fusion experiments:
[circle_fusion_model](https://vanderbilt.box.com/s/daknhqeow1fn2kg2tec4ncfzc46osdtg)

## CircleFusion demo
To run it on Whole Slide Images, please go to `circle_fusion` folder and run:
```
python /data/CircleNet/src/run_detection_for_scn.py circledet --circle_fusion --circle_fusion_eval --demo "/data/CircleNet/circle_fusion_test/WSI" --load_model_dir "/data/CircleNet/exp/circledet/cdf_models_new" --filter_boarder --demo_dir "/data/CircleNet/circle_fusion_test/result" --gt_dir "/data/CircleNet/circle_fusion_test/ground_truth" 
```
demo is the path to the WSI images that you want to process. load_model_dir is the path to your models' folder. demo_dir is where your result finally generate. gt_dir is the path to your ground truths,which are xml files here.
You can set them anywhere in your computer.

For testing the code，you can use the WSIs and ground truth here：
[circle_fusion_test](https://vanderbilt.box.com/s/qnnyo7ai97q9e7do6htc1rg6kcabqnlh)

There are a total of 10 models: five are older versions trained on 100,000 glomeruli, and the other five are newer versions that were additionally trained on an extra 100,000 glomeruli.
