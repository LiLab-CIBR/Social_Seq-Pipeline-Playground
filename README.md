# Social-Seq 项目：在示例数据集上运行 Social-Seq 处理代码流程
## 准备工作 
**数据集准备**

询问 Dr. liying 以获取数据集。

**环境准备**

确保安装了Social-Seq的深度学习环境。以便能够运行 `mmpose`, `mmdet (MASKRCNN)`, `DANNCE`, `SmoothNet`, `yolov8` 等基础深度学习模型。确保`lilab`, `multiview_calib` 工具包已经安装。可以参考Social-Seq的官方文档进行安装（准备中）。

## 1. 小球标定多相机系统
数据集为 `1-Ball_Calibration/2024-12-29_12-24-17.mp4`，使用 `mmpose` 关键点模型进行小球识别，然后使用`multiview_calib` 库进行标定多相机系统（外参）。多相机的OBS组成、内存标定。


【mmpose转tensorrt模型】组外测试前准备：将 mmpose 的小球检测深度学习模型从 onnx 转化为 tensorrt 模型，存放在'1-Ball_Calibration/mmpose_model/work_dirs/res50_coco_ball_512x320_cam9' 目录下，自动命名为`latest_full.engine`。详细脚本可见 '1-Ball_Calibration/covert_mmpose_model_to_tensorrt.sh'。

接下来可以开始标定多相机系统了。

```bash
bash 1-Ball_Calibration/p_calibration.sh 1-Ball_Calibration/2024-12-29_12-24-17.mp4 carl
```

其中 `carl` 是9相机对应的配置文件，包含每个相机的图像排列和相机内参。存放在`LILAB仓库包/lilab/cameras_setup/carl`目录下。

**组内快捷测试**：

```bash
PATH_BALL_MP4=/mnt/liying.cibr.ac.cn_usb2/social-seq-playground/1-Ball_Calibration/2024-12-29_12-24-17.mp4
SETUPNAME=carl

/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/multiview_scripts_dev/p_calibration.sh  $PATH_BALL_MP4  $SETUPNAME
```

结果生成：
- [可删] `FILE_keypoints.mp4`可视化视频.
- [重要保留] `FILE.calibpkl` 的多相机模型文件。
- [可删] `output/`多相机矫正的中间过程.


## 2. 社交三维姿态计算（SOCIAL）
进入项目：
```bash
cd 2-Social_3D_Pose/
```

**组内快捷测试**：

```bash
vdir=.
ball=/mnt/liying.cibr.ac.cn_usb2/social-seq-playground/2023-05-24_ball.calibpkl
config=/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9.py

python -m lilab.mmdet_dev_multi.s1_mmdet_videos2segpkl_dilate $vdir --pannels carl --config $config --maxlen 9000

python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask $vdir  # --maxlen 9000 check video  

ls $vdir/*.segpkl | xargs -n 1 -I {} -P 4 python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro {} --calibpkl "$ball"

# 绘制体素空间的视频（可选，不推荐）
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 8 -I {} python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video {} --vox_size 230  # check video
```

生成结果：
- [重要保留] `FILE.segpkl` 文件，包含分割后的二值掩码。
- [可删] `FILE_mask.mp4` 视频，可视化分割结果。
- [可删] `FILE_com3d_vol230.0.mp4` 视频，可视化体像素空间。

 请针对不同年龄的鼠，选择合适的 voxsize。

|         | Male | Female |
| ------- | ---- | ------ |
| DAY35   | 190  | 190    |
| DAY50   | 230  | 220    |
| DAY75   | 250  | 240    |
| DAY adult | 260  | 240    |


**接着开始关键点预测**：
```bash
cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young

# 1. Prepare data
# voxsize rat-black, voxsize rat-white, segpkl file
volsize_vfiles="
230 230 /mnt/liying.cibr.ac.cn_usb2/social-seq-playground/2-Social_3D_Pose/2023-05-24_14-43-58CBxDW_1min.segpkl
"

volsize_vfiles=$(echo "$volsize_vfiles" | grep -v '^[[:space:]]*$') #echo "$volsize_vfiles"
vfiles=$(echo "$volsize_vfiles" | awk '{print $3}')                 #echo "$vfiles"

# 2. Use DANNCE to predict 3D pose, nGPU=4
echo "$volsize_vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c 'python -m dannce.cli_trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size-list $1 $2 --video-file $3 --gpu-id $(($0%4))'
# xargs -P 2 means using 2 GPUs,配合 choosecuda 0,1,2,3 to confirm the number of GPUs used


# 3. Draw "unsmoothed" pose trajectories (optional)
# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 3'

# 4. Draw smoothed pose trajectories
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot_dzy
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 3 --postfix smoothed_foot '

# 5 Create 400p thumbnail video files (optional)
bash /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/openlabcluster_postprocess/create_400p.sh \
  /mnt/liying.cibr.ac.cn_usb2/social-seq-playground/2-Social_3D_Pose/
```

生成结果：

- [可删] `FILE.matcalibpkl` 文件，包含3D关键点预测结果。
- [保留] `FILE.smoothed_foot.matcalibpkl` 文件，SmoothNet 平滑关键点。
- [可删] `FILE_3_sktdraw_smoothed_foot.mp4` 视频，可视化平滑关键点。
- [可删] `FILE_1_400p.mp4` 视频，400x400缩略图视频。

## 3. 社交序列标签（SEQ）

```bash
conda activate OpenLabCluster
PROEJECT_DIR=/mnt/liying.cibr.ac.cn_usb2/social-seq-playground/3-Sequence_Labeling

# 动物在每个video里面算体长，算sniffzone
# 得到 'bodylength.pkl', 'rawfeat.pkl'
python -m lilab.lstm_bhv_bodylennorm_classify.s0_prepare_body_length $PROEJECT_DIR   
python -m lilab.lstm_bhv_bodylennorm_classify.s01_matcalibpkl2rawfeatpkl $PROEJECT_DIR

# 按速度 95% 分位数归一化
# 得到 rawfeat_norm.pkl
python -m lilab.lstm_bhv_bodylennorm_classify.s02_rawfeatpkl_to_norm $PROEJECT_DIR #--setspeed 0.05

# 从3D关键点出发，加上体长信息，算出行为分类
# --use-normed 使用速度归一化后的特征
# 得到 out_semiseq2seq_norm/lstm_offline.clippredpkl
python -m lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl_semiseq2seq $PROEJECT_DIR --use-normed 


# 只看镜像一致性的视频片段，representative
# 得到 out_semiseq2seq_norm/representitive_k36_filt_perc*/*.clippredpkl 和 .seqencepkl
conda activate mmdet
python -m lilab.OpenLabCluster_train.a6_clippredpkl_2_seqencepkl $PROEJECT_DIR/out_semiseq2seq_norm/*.clippredpkl

python -m lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl $PROEJECT_DIR/out_semiseq2seq_norm/lstm_offline.clippredpkl --already-mirrored
python -m lilab.OpenLabCluster_train.a6_clippredpkl_2_seqencepkl $PROEJECT_DIR/out_semiseq2seq_norm/representitive_k36_filt_perc*/*.clippredpkl --autoEnd

# 画出每个类别的视频片段 [可选]
python -m lilab.OpenLabCluster_train.a6b_clippredpkl_2_cluster400p  \
     $PROEJECT_DIR/out_semiseq2seq_norm/representitive_k36_filt_perc*/Representive_K36.clippredpkl \
     $PROEJECT_DIR
```

生成结果：

- [可删] `bodylength.pkl` 文件，从姿态关键点出发，计算每个视频每只大鼠的体长。
- [可删] `rawfeat.pkl` 从姿态关键点出发，计算社交特征(N=32维度，0.8秒时间窗）。
- [保留] `rawfeat_norm.mp4` 社交特征按体长归一化。
- [保留] `out_semiseq2seq_norm/*.clippredpkl` 文件，每一个片段得到监督标签。
- [保留] `out_semiseq2seq_norm/*_seqences.pkl` 文件，按video时序串联标签。
- [保留] `out_semiseq2seq_norm/out_semiseq2seq_norm/*` 文件夹，筛选出双重标签一致的片段（representative）。
- [可删] `out_semiseq2seq_norm/out_semiseq2seq_norm/3-Sequence_Labeling/out_semiseq2seq_norm/representitive_k36_filt_perc87/400p_clusters/clus_*.mp4` 视频，每个行为类别的视频，命名对应`cluster_names_k36.txt`，黑色鼠为视角中心。
