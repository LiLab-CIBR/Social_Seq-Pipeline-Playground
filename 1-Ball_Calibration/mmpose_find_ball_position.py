# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster  /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/SOLO/
# %%
import os
import os.path as osp
import numpy as np
import tqdm
import pickle
import ffmpegcv
import onnxruntime as ort
from lilab.cameras_setup import get_view_xywh_wrapper
import argparse


video_file = '/mnt/liying.cibr.ac.cn_usb2/social-seq-playground/1-ball_calibration/2024-12-29_12-24-17.mp4'
onnx_file = osp.join(osp.dirname(__file__), 'mmpose_model/work_dirs/res50_coco_ball_512x320_cam9/latest.full.onnx')


nrow, ncol = 3, 3
nview = 9
feat_h, feat_w = feat_shape = (320, 512) # feat all shape (9view, rgb, 320, 520)
cam_resolution_h, cam_resolution_w = (800, 1280)


#%% load model
def create_inference_session(onnx_file):
    """创建ONNX推理会话，优先使用GPU(CUDA)，否则使用CPU"""
    # 检查可用的计算 providers
    available_providers = ort.get_available_providers()
    print(f"可用的计算 providers: {available_providers}")
    
    # 优先选择CUDA（GPU），如果可用
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("使用GPU (CUDA) 进行推理")
    else:
        providers = ['CPUExecutionProvider']
        print("CUDA不可用，使用CPU进行推理")
    
    # 创建推理会话
    try:
        sess = ort.InferenceSession(onnx_file, providers=providers)
        return sess
    except Exception as e:
        print(f"创建推理会话失败: {e}")
        return None


def main(video_file, setupname):
    sess = create_inference_session(onnx_file)
    views_xywh = get_view_xywh_wrapper(setupname)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    output_name = sess.get_outputs()[0].name

    # %% load video
    vid = ffmpegcv.VideoCaptureNV(video_file, resize=(ncol*feat_shape[1], nrow*feat_shape[0]), resize_keepratio=False, pix_fmt='rgb24')
    nframe = len(vid)

    keypoints_xyp = np.zeros((nview, nframe, 1, 3), dtype=float)
    for iframe, frame in enumerate(tqdm.tqdm(vid)):
        # 将 frame (np.ndarray: 320xnrow, 512xncol, 3)按nrow, ncol 分割
        split_frames = frame.reshape(nrow, feat_h, ncol, feat_w, 3)  # 形状变为 (nrow, 子高, ncol, 子宽, 3)
        split_frames = split_frames.swapaxes(1, 2).reshape(ncol*nrow, feat_h, feat_w, 3)  # 交换轴，形状变为 (nrow * ncol, 子高, 子宽, 3)
        feat_NCHW = split_frames.transpose(0, 3, 1, 2).astype(np.float32)  # 交换轴，形状变为 (nrow * ncol, 3, 子高, 子宽)
        assert feat_NCHW.shape[1:] == tuple(input_shape[1:])
        
        # 获取热图
        heat_maps_NKHW = sess.run([output_name], {input_name: feat_NCHW})[0]  #(9,1,320,512)
        heat_maps_NHW = heat_maps_NKHW[:,0]  #(9,320,512)

        # get the peak position and peak value of heat_maps_NHW in each batch(N)
        batch_size, h, w = heat_maps_NHW.shape
        for iview, heat_map in enumerate(heat_maps_NHW):
            max_idx = np.argmax(heat_map)
            y = max_idx // w
            x = max_idx % w
            keypoints_xyp[iview, iframe, 0] = [x, y, heat_map[y, x]]

    keypoints_xyp[..., 0] *= cam_resolution_w / feat_w
    keypoints_xyp[..., 1] *= cam_resolution_h / feat_h

    info = {'vfile': video_file, 'nview': len(views_xywh), 'fps':  vid.fps}
    outpkl = os.path.splitext(video_file)[0] + '.matpkl'
    outdict = dict(
        keypoints = keypoints_xyp,
        views_xywh = views_xywh,
        info = info
    )
    pickle.dump(outdict, open(outpkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('video_file', type=str, help='input video file')
    parser.add_argument('--pannels', type=str, help='setupname')
    args = parser.parse_args()
    main(args.video_file, args.pannels)
