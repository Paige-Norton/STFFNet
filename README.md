# STFFNet
**[ICPR2024]** [A Novel Encoder-Decoder Network with Multi-domain Information Fusion for Video Deblurring]()

by [Peiqi Xie](https://github.com/Paige-Norton), Jinhong He, Chengyun Song, and Minglong Xue⋆

<!-- This work presents an efficient RNN-based model and **the first real-world dataset for image/video deblurring** :-) -->

## Visual Results

### Results on BSD (Real-world)
![image](https://github.com/Paige-Norton/STFFNet/doc/compare.png)

Quantitative results on different setups of BSD:

<img src="https://github.com/Paige-Norton/STFFNet/doc/BSD_results.png" alt="bsd_config" width="800"/>


### Results on REDS and GOPRO (Synthetic)
<!-- ![image](https://github.com/zzh-tech/Images/blob/master/ESTRNN/reds.gif) -->

Quantitative results on different setups of REDS and GOPRO:

<img src="https://github.com/Paige-Norton/STFFNet/doc/other_results.png" alt="bsd_config" width="800"/>


## Quick Start

### Prerequisites

- Python 3.6
- PyTorch 1.6 with GPU
- opencv-python
- scikit-image
- lmdb
- thop
- tqdm
- tensorboard

### Downloading Datasets

Please download and unzip the dataset file for each benchmark.

- [**BSD**](https://drive.google.com/drive/folders/1LKLCE_RqPF5chqWgmh3pj7cg-t9KM2Hd?usp=sharing) ([**Full BSD with RAW**](https://drive.google.com/file/d/1VJdyojIAriC5QZp2N_0umEqkIMk1_9HA/view?usp=sharing))
- [GOPRO](https://drive.google.com/file/d/1dHJX-TIY-ZsSV6-PbPZzmockp1H3B_5w/view?usp=sharing)
- [REDS](https://drive.google.com/file/d/1lFHndopTiAAOIEkjZdvrziA8p17y4rjD/view?usp=sharing)

If you failed to download BSD from Google drive, please try the following BaiduCloudDisk version:  
[BSD 1ms8ms](https://pan.baidu.com/s/1i7iMOZVOvBWmNYi8zkQIpw), password: bsd1  
[BSD 2ms16ms](https://pan.baidu.com/s/1ur-XHeNoSTPFQJwBVfbofQ), password: bsd2  
[BSD 3ms24ms](https://pan.baidu.com/s/1QNJlxiduwbQzCypy-7Mlbw), password: bsd3  


### Training

Specify *\<path\>* (e.g. "*./dataset/*") as where you put the dataset file.

Modify the corresponding dataset configurations in the command, or change the default values in "*./para/paramter.py*". 

Training command is as below:

```bash
python main.py --data_root <path> --dataset BSD --ds_config 2ms16ms
```

You can also tune the hyper-parameters such as batch size, learning rate, epoch number (P.S.: the actual batch size for ddp mode is num_gpus*batch_size):

```bash
python main.py --lr 1e-4 --batch_size 4 --num_gpus 2 --trainer_mode ddp
```

If you want to train on your own dataset, please refer to "*/data/how_to_make_dataset_file.ipynb*".

### Inference

Please download [checkpoints](https://drive.google.com/file/d/1w68kAw56tGCjG4M96_zYmls8fQaTH1RM/view?usp=sharing) of pretrained models for different setups and unzip them under the main directory.

#### Dataset (Test Set) Inference

Command to run a pre-trained model on BSD (3ms-24ms):

```bash
python main.py --test_only --test_checkpoint ./checkpoints/ESTRNN_C80B15_BSD_3ms24ms.tar --dataset BSD --ds_config 3ms24ms --video
```

#### Blurry Video Inference

Specify **"--src \<path\>"** as where you put the blurry video file (e.g., "--src ./blur.mp4") or video directory (e.g., "--src ./blur/", the image files under the directory should be indexed as "./blur/00000000.png", "./blur/00000001.png", ...).

Specify **"--dst \<path\>"** as where you store the results (e.g., "--dst ./results/").

Command to run a pre-trained model for a blurry video is as below:

```bash
python inference.py --src <path> --dst <path> --ckpt ./checkpoints/ESTRNN_C80B15_BSD_2ms16ms.tar
```

<!-- ## Citing

If you use any part of our code, or ESTRNN and BSD are useful for your research, please consider citing:

```bibtex
@inproceedings{zhong2020efficient,
  title={Efficient spatio-temporal recurrent neural network for video deblurring},
  author={Zhong, Zhihang and Gao, Ye and Zheng, Yinqiang and Zheng, Bo},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part VI 16},
  pages={191--207},
  year={2020},
  organization={Springer}
}
``` -->
