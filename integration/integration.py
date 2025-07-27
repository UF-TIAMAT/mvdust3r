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
import open3d as o3d

import matplotlib.pyplot as pl
from dust3r.inference import inference, inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

inf = np.inf

def get_reconstructed_scene(model, device, silent, image_size, filelist, min_conf_thr, n_frame):
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
    
    mvdust3r_output = inference_mv(imgs, model, device, verbose=not silent)

    mvdust3r_output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
    for x, img in zip(mvdust3r_output['pred2s'], imgs[1:]):
        x['rgb'] = img['img'].permute(0,2,3,1)

    renderer, cams2world = get_rendering_from_scene(mvdust3r_output, min_conf_thr=min_conf_thr)


    for i, cam in enumerate(cams2world):
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        camera_matrix = np.linalg.inv(cams2world[i] @ OPENGL @ rot)

        position = camera_matrix[:3, 3]
        forward = camera_matrix[:3, 2]
        up = camera_matrix[:3, 1]
        lookat = position + forward

        renderer.scene.camera.look_at(
            center=lookat,    # look at origin
            eye=position,       # camera position
            up=up              # up vector
            )    
        image = renderer.render_to_image()
        o3d.io.write_image(f"rendering_results/mesh_results/cam_{i}_hori_rendered_image_mesh_newx_finally_fu.png", image)

    print()


def get_rendering_from_scene(output, min_conf_thr=3):
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

        assert len(pts3d) == len(msk) <= len(rgbimg) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(rgbimg)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], msk[i]))
        mesh = cat_meshes(meshes)

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # or "defaultLit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]  # White base color
        material.point_size = 3.0


        faces_3 = np.zeros_like(mesh['faces'])
        vertices_3 = np.zeros((len(mesh['faces']) * 3, 3), dtype=np.float32)

        for index_face, face in enumerate(mesh['faces']):
            index_vertex = index_face * 3
            vertices_3[index_vertex] = mesh['vertices'][face[0]]
            vertices_3[index_vertex + 1] = mesh['vertices'][face[1]]
            vertices_3[index_vertex + 2] = mesh['vertices'][face[2]]
            faces_3[index_face] = np.arange(index_vertex, index_vertex + 3)

        colors = np.repeat(mesh['face_colors'], 3, axis=0)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_3)
        mesh.triangles = o3d.utility.Vector3iVector(faces_3)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # or "defaultLit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]  # White base color


        renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
        renderer.scene.add_geometry("mesh", mesh, material)
        camera = renderer.scene.camera
        camera.set_projection(field_of_view=79.0, aspect_ratio=640/480, near_plane=0.01, far_plane=1000.0, field_of_view_type=camera.FovType.Horizontal)

    return renderer, cams2world


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

    print("Reached the end")

    get_reconstructed_scene(
        model=model,
        device=device,
        silent=False,
        image_size=224,  # This should 224 always.
        filelist=filelist,  # Replace with your image paths
        min_conf_thr=0.5,
        n_frame=2
    )


#   for i in range(-5, 5):
#         for j in range(-5, 5):
renderer.scene.camera.look_at(
    center=lookat,    # look at origin
    eye=position,       # camera position
    up=up              # up vector
    )    
image = renderer.render_to_image()
o3d.io.write_image(f"rendering_results/mesh_results/2rendered_image_mesh_newx_finally_fu.png", image)