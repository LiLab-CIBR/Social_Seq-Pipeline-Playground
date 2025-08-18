cd /mnt/liying.cibr.ac.cn_usb2/social-seq-playground/1-ball_calibration/mmpose_model
mfile='res50_coco_ball_512x320_cam9.py'
mfile_nake=${mfile%.*}

# [ESCAPE] MMPOSE latest.pth -> latest_full.onnx
# python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic


# [CONVERT] latest_full.onnx -> latest_full.engine
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
    --fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full.engine \
    --timingCacheFile=work_dirs/${mfile_nake}/.cache.txt \
    --workspace=3072 --optShapes=input_1:4x3x320x512 \
    --minShapes=input_1:1x3x320x512 --maxShapes=input_1:10x3x320x512
