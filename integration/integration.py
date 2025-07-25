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
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as pl
from dust3r.inference import inference, inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

from gsplat.rendering import rasterization

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
    # input('press enter to continue')

    # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
    output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
    for x, img in zip(output['pred2s'], imgs[1:]):
        x['rgb'] = img['img'].permute(0,2,3,1)

    outdir = "data_test/out_dir"

    output = get_3D_model_from_scene(outdir, silent, output, min_conf_thr, as_pointcloud, transparent_cams, cam_size)

    print("Hello!, done running")
       
    means = torch.stack(output['means'], 0).reshape(-1, 3).cuda()
    quats = torch.stack(output['quats'], 0).reshape(-1, 4).cuda()
    scales = torch.stack(output['scales'], 0).reshape(-1, 3).cuda()
    opacities = torch.stack(output['opacities'], 0).reshape(-1).cuda()
    colors = torch.stack(output['colors'], 0).flatten(0, -2).cuda()

    sh_base = colors.shape[-1] // 3
    colors = colors.reshape(-1, sh_base, 3)

    viewmats = output["viewmats"].cuda()
    intrinsics = output['intrinsics'].cuda()
    intrinsics = intrinsics.repeat(viewmats.shape[0], 1, 1)  # Repeat intrinsics for each camera

    n_cam = viewmats.shape[0]
    
    width = 640
    height = 480
    znear = 0.1
    zfar = 1000.0
    bg_color = torch.ones((1, 3)).repeat(n_cam, 1).to(device)  # Background color for rasterization
    eps2d = 0.3
    sh_base = colors.shape[-1]//3
    sh_degree = int(np.sqrt(sh_base)) - 1
    
    out_img, out_alpha, _ = rasterization(
        means=means, 
        quats=quats, 
        scales=scales, 
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=znear,
        far_plane=zfar,
        backgrounds=bg_color,
        sh_degree=sh_degree,
        eps2d=eps2d,
    )

    print("finished rasterization")
    
def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(outdir, silent, output, min_conf_thr=3, as_pointcloud=False, transparent_cams=False, cam_size=0.05, only_model=False):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """

    with torch.no_grad():
        
        _, h, w = output['pred1']['rgb'].shape[0:3] # [1, H, W, 3]
        rgbimg = [output['pred1']['rgb'][0]] + [x['rgb'][0] for x in output['pred2s']]
        for i in range(len(rgbimg)):
            rgbimg[i] = (rgbimg[i] + 1) / 2
        pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
        conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0) # [N, H, W]
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres
        
        # calculate focus:

        conf_first = conf[0].reshape(-1) # [bs, H * W]
        conf_sorted = conf_first.sort()[0] # [bs, h * w]
        conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
        valid_first = (conf_first >= conf_thres) # & valids[0].reshape(bs, -1)
        valid_first = valid_first.reshape(h, w)

        focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()

        intrinsics = torch.eye(3,)
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        focals = torch.Tensor([focals]).reshape(1,).repeat(len(rgbimg))

        
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2]
        
        c2ws = []
        for (pr_pt, valid) in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None])
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]
        focals = to_numpy(focals)

        pts3d = to_numpy(pts3d)
        msk = to_numpy(msk)

    glb_file = _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    conf = to_numpy([x[0] for x in conf.split(1, dim=0)])
    rgbimg = to_numpy(rgbimg)

    means = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
    quats = [output['pred1']['rotation'][0]] + [x['rotation'][0] for x in output['pred2s']]
    scales = [output['pred1']['scale'][0]] + [x['scale'][0] for x in output['pred2s']]
    opacities = [output['pred1']['opacity'][0]] + [x['opacity'][0] for x in output['pred2s']]
    colors = [output['pred1']['rgb'][0]] + [x['rgb'][0] for x in output['pred2s']]

    output = {}
    output['means'] = means
    output['quats'] = quats
    output['scales'] = scales
    output['opacities'] = opacities
    output['colors'] = colors
    output['viewmats'] = cams2world
    output['intrinsics'] = intrinsics   

    # if only_model:
    #     return glb_file
    # return glb_file, rgbimg, conf, cams2world 

    return output

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
        image_size=224,  # This should 224 always.
        filelist=filelist,  # Replace with your image paths
        min_conf_thr=0.5,
        as_pointcloud=False,
        transparent_cams=True,
        cam_size=0.05,
        n_frame=2
    )


