import argparse
import copy
import functools
import math
import os
import tempfile
from copy import deepcopy

import gradio
import numpy as np
import torch
import trimesh

import matplotlib.pyplot as pl
from dust3r.inference import inference, inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

inf = np.inf

def get_reconstructed_scene(model, device, silent, image_size, filelist, min_conf_thr,
                            as_pointcloud, transparent_cams, cam_size, n_frame):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent, n_frame = n_frame)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    for img in imgs:
        img['true_shape'] = torch.from_numpy(img['true_shape']).long()

    if len(imgs) < 12:
        if len(imgs) > 3:
            imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
        if len(imgs) > 6:
            imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
    else:
        change_id = len(imgs) // 4 + 1
        imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
        change_id = (len(imgs) * 2) // 4 + 1
        imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
        change_id = (len(imgs) * 3) // 4 + 1
        imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])
    
    output = inference_mv(imgs, model, device, verbose=not silent)
    input('press enter to continue')

    # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
    output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
    for x, img in zip(output['pred2s'], imgs[1:]):
        x['rgb'] = img['img'].permute(0,2,3,1)

    print("Hello!, done running")


if __name__ == "__main__":

    # Define this to change weight path
    weights_path = "checkpoints/MVD.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "MVDp" in weights_path:
        model_name = "MVDp"
    elif "MVD" in weights_path:
        model_name = "MVD"

    if model_name == "MVD":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
        model.to(device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    elif model_name == "MVDp":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True, n_ref = 4)
        model.to(device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    

    # NOTE: filelist to debug
    filelist = ["data_test/path_images/00000.png", "data_test/path_images/00001.png", "data_test/path_images/00002.png"]  # Replace with your image paths

    get_reconstructed_scene(
        model=model,
        device=device,
        silent=False,
        image_size=(224, 224), # This should (224, 224) always. 
        filelist=filelist,  # Replace with your image paths
        min_conf_thr=0.5,
        as_pointcloud=False,
        transparent_cams=True,
        cam_size=0.05,
        n_frame=2
    )