# SkyPose：Skeleton Consistency Guided Diffusion Model for 3D Human Pose Estimation




<img width="16384" height="9374" alt="bp-denosier" src="https://github.com/user-attachments/assets/3aaaa05e-1bd5-40e2-bd17-3ede679d2d27" />


## Dataset

### Human3.6M

#### Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d', or direct download our processed data [here](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing) and unzip it.
2. Slice the motion clips by running the following python code in `tools/convert_h36m.py`:

```
python convert_h36m.py
```

### MPI-INF-3DHP

#### Preprocessing

Please refer to - [MotionAGFormer](https://github.com/taatiteam/motionagformer) for dataset setup.


The dynamic changes observed during the denoising process can also be viewed online at [here](https://www.bilibili.com/video/BV1ZY19BbEWy/?spm_id_from=333.1387.upload.video_card.click&vd_source=e1fb30b1323d74d327f07f88788b4ef9) and [here](https://www.bilibili.com/video/BV1qY19BbERN/?spm_id_from=333.1387.upload.video_card.click&vd_source=e1fb30b1323d74d327f07f88788b4ef9), or download the video from the repository.





