# BGA-Net: A Bézier Global Attention Lane Detection Network

Our work is based on [pytorch auto drive](https://github.com/voldemortX/pytorch-auto-drive)
```Shell
BGA-Net/

├── configs/
│   └── lane_detection/
│       └── bezierlanenet/
│           └── resnet18_culane_aug1b_with_transformer.py  # Main configuration file
├── utils/
│   ├── models/
│   │   └── lane_detection/
│   │       └── bezier_lane_net.py    # Backbone & Transformer architecture
│   └── losses/
│       └── CombineLoss.py            # Multi-task loss function
├── data/                             # Datasets (CULane, TuSimple, etc.)
├── tools/                            # Training & evaluation scripts
├── docs/
└── README.md
```
# Key Components

1.Configuration file: configs/lane_detection/bezierlanenet/resnet18_culane_aug1b_with_transformer.py Defines model architecture including Backbone settings, Attention Module settings, Head settings, data augmentation settings, optimizer settings, and training hyperparameters for CULane dataset.

2.Model Architecture file: utils/models/lane_detection/bezier_lane_net.py Defines Backbone: ResNet-18 for hierarchical feature extraction; Attention Module: Transformer-based global context aggregator; Head: Bézier curve parameter decoder for lane representation.

3.Loss Function file: configs/lane_detection/common/optims/combineloss.py Defines combines multiple supervision signals: Bézier mean curve loss function; Lane loss function.

# Clone the framework
```Shell
git clone https://github.com/liweiac/BGA-Net.git
cd BGA-Net
```
# Install dependencies
```Shell
pip install -r requirements.txt
```
# Training
```Shell
python main_landet.py --train --config=/01080005/pytorch-auto-drive-master/pytorch-auto-drive-master/configs/lane_detection/bezierlanenet/resnet18_culane_aug1b_with_transformer.py 
```
# Evaluation
```Shell
python tools/test.py --config configs/lane_detection/bezierlanenet/resnet18_culane_aug1b_with_transformer.py --weight-path weights/bga_net_culane_best.pth
```
# Visualization example
```Shell
python tools/vis/lane_img_dir.py --image-path=/01080005/Lanedatasets/culane/driver_100_30frame/05251408_0410.MP4 --gt-keypoint-path=/01080005/Lanedatasets/culane/driver_100_30frame/05251408_0410.MP4  --image-suffix=.jpg --gt-keypoint-suffix=.lines.txt --save-path=PAD_test_images/lane_test_images/culane_gt_compare --config=/01080005/pad/pad/configs/lane_detection/bezierlanenet/resnet18_culane_aug1b_with_transformer --style=point --mixed-precision --pred
```
# Acknowledgements

PyTorch Auto Drive: Foundational codebase

CULane Dataset: Benchmark dataset

All contributors and the autonomous driving research community
