# Learning Character-Agnostic Motion for Motion Retargeting in 2D

We provide PyTorch implementation for our paper _Learning Character-Agnostic Motion for Motion Retargeting in 2D_.

(put gifs here)



## Prerequisites

xxx



## Getting Started

### Installation

- Clone this repo

  ```
  xxx
  ```

- Install dependencies

  xxx



### Run demo examples

We provide pretrained models and several video examples, along with their OpenPose outputs. 

- Run the full model to combine motion, skeleton, view angle from three inputs video:

  ```
  python predict.py -n full --model_path ./model/pretrained_full.pth -v1 ./examples/tall_man -v2 ./examples/midget -v3 ./examples/model -h1 720 -w1 720 -h2 720 -w2 720 -h3 720 -w3 720 -o ./outputs/full-demo
  ```

  Results will be saved in `./outputs/full-demo`.

- Run the two encoder model to transfer motion and skeleton between two inputs video:

  ```
  python predict.py -n skeleton --model_path ./model/pretrained_skeleton.pth -v1 ./examples/tall_man -v2 ./examples/midget -h1 720 -w1 720 -h2 720 -w2 720 -o ./outputs/skeleton-demo
  ```

- Run the two encoder model to transfer motion and view angle between two inputs video:

  ```
  python predict.py -n view --model_path ./model/pretrained_view.pth -v1 ./examples/tall_man -v2 ./examples/model -h1 720 -w1 720 -h2 720 -w2 720 -o ./outputs/full-demo
  ```



### Run with your own video

To run our models with your own video, you need first to run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) on it and then supply the resulting JSON directory as above.



## Train from scratch

### Prepare Data

- Download Mixamo Data

  For the sake of convenience, we pack the Mixamo Data that we use. To download it, see [Google Drive]() or [Baidu Drive](). After downloading, please put it under `./mixamo_data`.

  > NOTE: Our Mixamo dataset only covers a part of the whole collections provided by the Mixamo website. If you want to collect Mixamo Data by yourself, you can follow the our guide [here](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/dataset/Guide%20For%20Downloading%20Mixamo%20Data.md).

- Preprocess the downloaded data

  ```
  python ./dataset/preprocess.py
  ```



### Train

- Train the full model (with three encoders):

```
python train.py -n full -g 1
```

(more info about arguments)



## Citation

xxx