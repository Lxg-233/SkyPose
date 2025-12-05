# SkyPose：Skeleton Consistency Guided Diffusion Model for 3D Human Pose Estimation




<img width="16384" height="9374" alt="bp-denosier" src="https://github.com/user-attachments/assets/3aaaa05e-1bd5-40e2-bd17-3ede679d2d27" />




The dynamic changes observed during the denoising process can also be viewed online at [here](https://www.bilibili.com/video/BV1ZY19BbEWy/?spm_id_from=333.1387.upload.video_card.click&vd_source=e1fb30b1323d74d327f07f88788b4ef9) and [here](https://www.bilibili.com/video/BV1qY19BbERN/?spm_id_from=333.1387.upload.video_card.click&vd_source=e1fb30b1323d74d327f07f88788b4ef9), or download the video from the repository.



## Dataset setup
Please download the dataset here and refer to VideoPose3D to set up the Human3.6M dataset ('./dataset' directory).

```
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Test the model



To test on Human3.6M on single frame, run:

```
python main2.py 
```



## Train the model



To train on Human3.6M with single frame, run:

```
python main2.py --train -n 'name'
```

## Acknowledgement



Our code is extended from the following repositories. We thank the authors for releasing the codes.

- [HTnet](https://github.com/vefalun/HTNet)
- [D3DP](https://github.com/paTRICK-swk/D3DP)

