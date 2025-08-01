from typing import Any, Dict
import time
import os
import random
import socket
import numpy as np
import requests
import cv2
import base64
from flask import Flask, jsonify, request, Response
import torch
import importlib
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import pickle
import torchvision.transforms as tvf
import copy
from copy import deepcopy
import open3d as o3d
from scipy.spatial.transform import Rotation


import matplotlib.pyplot as pl
from dust3r.inference import inference, inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

from similarity_transform import kabsch_umeyama

inf = np.inf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError
    
def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = pickle.loads(request.data)
        response = model.process_payload(payload)
        response = pickle.dumps(response)

        return Response(response, content_type="application/octet-stream")

    app.run(host="localhost", port=port, debug=False)

def send_request(url: str, payload: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, payload=payload)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response

def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

def _send_request(url: str, payload: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        # payload = {}
        # for k, v in kwargs.items():
        #     if isinstance(v, np.ndarray):
        #         payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
        #     else:
        #         payload[k] = v

        # Set the headers
        headers={'Content-Type': 'application/octet-stream'}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, data=payload)
                if resp.status_code == 200:
                    result = pickle.loads(resp.content)
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result

class MVDust3RModel:

    def __init__(self, weights_path: str = "checkpoints/MVD.pth"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if "MVDp" in weights_path:
            model_name = "MVDp"
        elif "MVD" in weights_path:
            model_name = "MVD"

        if model_name == "MVD":
            model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
            model.to(self.device)
            model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(self.device)
            state_dict_loaded = model_loaded.state_dict()
            model.load_state_dict(state_dict_loaded, strict=True)
        elif model_name == "MVDp":
            model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True, n_ref = 4)
            model.to(self.device)
            model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(self.device)
            state_dict_loaded = model_loaded.state_dict()
            model.load_state_dict(state_dict_loaded, strict=True)

        self.model = model
        self.model.eval()

    @staticmethod
    def _resize_pil_image(img, long_edge_size):
        S = max(img.size)
        if S > long_edge_size:
            interp = PIL.Image.LANCZOS
        elif S <= long_edge_size:
            interp = PIL.Image.BICUBIC
        new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
        return img.resize(new_size, interp)

    @staticmethod
    def _load_images(img_arr, size, square_ok=False, verbose=True):
    # Resize and crop images

        imgs = []
        imgs_raw = []

        if verbose:
            print(f'>> Loading a list of {len(img_arr)} numpy image arrays')

        imgs_raw = [Image.fromarray(arr) for arr in img_arr]

        for img in imgs_raw:
            W1, H1 = img.size
            if size == 224:
                img = MVDust3RModel._resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                img = MVDust3RModel._resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not square_ok and W == H:
                    halfh = 3*halfw//4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

            W2, H2 = img.size
            imgs.append(dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs))
            ))

        assert imgs, 'No images found.'
        if verbose:
            print(f' (Found {len(imgs)} images)')

        return imgs


    def _get_rendering_from_scene(self, output, min_conf_thr=3):
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

            current_time = time.strftime("%d-%H%M%S")
            save_path = f"/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/mesh/{current_time}.ply"
            o3d.io.write_triangle_mesh(save_path, mesh)

            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"  # or "defaultLit"
            material.base_color = [1.0, 1.0, 1.0, 1.0]  # White base color


            renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
            renderer.scene.add_geometry("mesh", mesh, material)
            camera = renderer.scene.camera

            # TODO: Make this parametrizable
            camera.set_projection(field_of_view=79.0, aspect_ratio=640/480, near_plane=0.01, far_plane=1000.0, field_of_view_type=camera.FovType.Horizontal)

        return renderer, cams2world


    def get_reconstructed_scene(self, model, device, silent, image_size, img_arr, min_conf_thr):
        """
        from a list of images, run dust3r inference, global aligner.
        then run get_3D_model_from_scene
        """

        img_ids = []
        imgs = self._load_images(img_arr=img_arr, size=image_size, verbose=not silent)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            img_ids = [0, 0]
        for img in imgs:
            img['true_shape'] = torch.from_numpy(img['true_shape']).long()

        if len(imgs) < 12:
            img_ids = [i for i in range(len(imgs))]
            if len(imgs) > 3:
                imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
                img_ids[1], img_ids[3] = img_ids[3], img_ids[1]
            if len(imgs) > 6:
                imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
                img_ids[2], img_ids[6] = img_ids[6], img_ids[2]
        else:
            img_ids = [i for i in range(len(imgs))]
            change_id = len(imgs) // 4 + 1
            imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
            img_ids[1], img_ids[change_id] = img_ids[change_id], img_ids[1]

            change_id = (len(imgs) * 2) // 4 + 1
            imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
            img_ids[2], img_ids[change_id] = img_ids[change_id], img_ids[2]

            change_id = (len(imgs) * 3) // 4 + 1
            imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])
            img_ids[3], img_ids[change_id] = img_ids[change_id], img_ids[3]

        print(len(imgs), "images loaded for inference")
        
        mvdust3r_output = inference_mv(imgs, model, device, verbose=not silent)

        mvdust3r_output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
        for x, img in zip(mvdust3r_output['pred2s'], imgs[1:]):
            x['rgb'] = img['img'].permute(0,2,3,1)

        renderer, cams2world = self._get_rendering_from_scene(mvdust3r_output, min_conf_thr=min_conf_thr)

        

        # initial = np.eye(4, dtype=np.float32)

        # + Z is the forward
        # - Y is the up
        # 180 degree rotation around X or Y axis?? 

        # center = np.array([0, 0, 1], dtype=np.float32)
        # eye = np.array([0, 0, 0], dtype=np.float32)
        # NOTE: important. (0, -1, 0) is the up vector in OpenGL
        # up = np.array([0, -1, 0], dtype=np.float32)

        # renderer.scene.camera.look_at(
        #     center=center,    # look at origin
        #     eye=eye,       # camera position
        #     up=up              # up vector
        #     )    
        # image = renderer.render_to_image()
        # o3d.io.write_image(f"/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/server/{len(img_arr)}_+45.png", image)

        return renderer, cams2world, img_ids 


    def generate_novel_views(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        print("Generating novel views...")
        # payload = pickle.loads(payload)
        input_img_arr = payload["rgb_cache"]["images"]
        input_camera_matrix = payload["rgb_cache"]["transforms"]
        target_camera_matrix = payload["transformation_matrix"]


        renderer, cams2world, img_ids = self.get_reconstructed_scene(
            model=self.model,
            device=self.device,
            silent=True,
            image_size=224,
            img_arr=input_img_arr,
            min_conf_thr=0.5
        )

        # input_translations = []
        # output_translations = []

        # for i, img in enumerate(input_camera_matrix):
        #     input_translations.append(img[:3, 3])
        #     output_translations.append(cams2world[img_ids.index(i)][:3, 3])

        # R, s, t = kabsch_umeyama(np.array(input_translations[-1]), np.array(output_translations[-1]))


        # rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # world_2_world_transform = np.linalg.inv(input_camera_matrix[0])
        # rot = np.eye(4)
        # rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()

        # flip = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

        # flip_y_and_z = np.zeros((4, 4), dtype=np.float32)
        # flip_y_and_z[0, 0] = 1
        # flip_y_and_z[1, 2] = 1
        # flip_y_and_z[2, 1] = 1
        # flip_y_and_z[3, 3] = 1

        # transformed_camera_matrix = world_2_world_transform @ target_transform

        # f.write(f"Target camera {i}:\n")
        # f.write(f"World2World_transform: {world_2_world_transform}\n")
        # f.write(f"Target transform:{target_transform}\n")
        # f.write(f"Transformed Camera Matrix: {transformed_camera_matrix}\n")
        # # f.write(f"Rotation:\n{rot}\n")
        # f.write(f"OPENGL:\n{OPENGL}\n")
        # f.write(f"UP: {transformed_camera_matrix[:3, 1]}\n")
        # f.write(f"LOOK AT: {transformed_camera_matrix[:3, 3]}\n")
        # f.write(f"CAMERA POSITION: {transformed_camera_matrix[:3, 3] + transformed_camera_matrix[:3, 2]}\n")
        # f.write("\n")

        # rendered_imgs = []
        # with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/axis_rotation/debug/matrixes.txt", "w") as f:

        r_0 = Rotation.from_matrix(input_camera_matrix[0][:3, :3])
        reference_yaw, reference_pitch, reference_roll = r_0.as_euler('zxy')
        reference_position = input_camera_matrix[0][:3, 3]
        rendered_imgs = []  

        # with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/axis_rotation/debug.log", "w") as f: 

        for i, target_transform in enumerate(target_camera_matrix):
            
            # f.write(f"Target camera {i}:\n")
            # f.write(f"Target transform: {target_transform}\n")

            r = Rotation.from_matrix(target_transform[:3, :3])
            yaw, pitch, roll = r.as_euler('zxy')

            # f.write(f"Yaw, Pitch, Roll: {yaw}, {pitch}, {roll}\n")

            yaw_diff = yaw - reference_yaw
            forward = np.array([np.cos(yaw_diff + np.pi/2), 0, np.sin(yaw_diff + np.pi/2)], dtype=np.float32)

            # f.write(f"Yaw-diff-forward: {yaw}, {yaw_diff}, {forward}\n")

            position = target_transform[:3, 3] - reference_position
            position[1], position[2] = position [2], position[1]  # Swap Y and Z for OpenGL

            # f.write(f"Position: {position}\n")

            # Apply the transformation to the camera
            renderer.scene.camera.set_projection(
                field_of_view=79.0, 
                aspect_ratio=640/480, 
                near_plane=0.01, 
                far_plane=1000.0, 
                field_of_view_type=renderer.scene.camera.FovType.Horizontal
            )
            renderer.scene.camera.look_at(
                center=position + forward,  # look at the camera position
                eye=position ,  # camera position
                up=[0, -1, 0]  # up vector
            )
            

            image = renderer.render_to_image()
            o3d.io.write_image(f"/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/axis_rotation/debug_2/target_{len(input_img_arr)}_{i}.png", image)
            rendered_imgs.append(np.asarray(image))

        return {"response": rendered_imgs}

class MVDust3RModelClient:
    def __init__(self, port: int = 12400):
        self.url = f"http://localhost:{port}/mvdust3r"

    def generate_novel_views(self, payload) -> Dict[str, Any]:
        print(f"MVDust3RModelClient.generate_novel_views:")
        response = send_request(self.url, payload=payload)
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12400)
    parser.add_argument("--weights_path", type=str, default="checkpoints/MVD.pth")
    args = parser.parse_args()

    print("Loading model...")

    class MVDust3RModelServer(ServerMixin, MVDust3RModel):
        def process_payload(self, payload: dict) -> dict:
            print(f"MVDust3RModelServer.process_payload:")
            response = self.generate_novel_views(payload=payload)
            return response


    mv3dust3r = MVDust3RModelServer(weights_path=args.weights_path)
    print("Model loaded successfully.")
    print(f"Hosting model on port {args.port}...")
    host_model(mv3dust3r, name="mvdust3r", port=args.port)