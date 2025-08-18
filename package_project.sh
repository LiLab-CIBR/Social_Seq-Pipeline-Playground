#!/bin/bash

# RELEASE: DATA and CODE
# 设置zip文件名，包含当前日期
DATE=$(date +%Y%m%d)
ZIP_NAME="Pipeline_Playground_DATA_CODE_with_DATA_${DATE}.zip"


# 创建zip文件，包含指定文件和文件夹，同时排除不需要的文件
zip -r "$ZIP_NAME" \
  1-Ball_Calibration/ \
  2-Social_3D_Pose/ \
  3-Sequence_Labeling/ \
  -x "*/result/*" \
     "*/__pycache__/*" \
     "*.DS_Store"

echo "项目已打包为 $ZIP_NAME"