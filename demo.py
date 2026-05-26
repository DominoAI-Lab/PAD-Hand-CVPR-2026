import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import subprocess
import tempfile

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DEMO_DIR)

import torch
import cv2
import numpy as np
import argparse
import trimesh
import pyrender
from tqdm import tqdm

from models.pad_hand import PAD_Hand
from models.MANO import MANO
from models.difussionsampler import DiffusionSampler
from models.utils import rotmat2eulerzxy, batch_euler2matzxy

LIGHT_BLUE  = (0.0, 0.278, 0.671)
LIGHT_GREEN = (0.0, 209/255, 202/255)


def load_pad_hand_model(checkpoint_path, device):
    model = PAD_Hand(
        device=device, seqlen=16, decoder=True,
        d_model=512, nhead=8, d_hid=512, nlayers=4, dropout=0.1,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.mode = 'test'
    return model


def run_wilor_inference(video_path):
    """Run wilor_inference.py in the wilor conda env, return path to saved .npz."""
    tmp = tempfile.NamedTemporaryFile(suffix='_wilor.npz', delete=False)
    tmp.close()
    npz_path = tmp.name
    script   = os.path.join(_DEMO_DIR, 'wilor_inference.py')
    cmd = ['conda', 'run', '-n', 'hamer', 'python', script,
           '--video', video_path, '--output', npz_path]
    print('Running WiLoR inference in wilor env...')
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f'wilor_inference.py failed with exit code {ret}')
    return npz_path


def load_wilor_results(npz_path):
    """Load per-frame WiLoR predictions from .npz; return frame_preds list and fps."""
    data = np.load(npz_path)
    fps  = float(data['fps'][0])
    n    = data['vertices'].shape[0]

    frame_preds = []
    for i in range(n):
        if np.isnan(data['vertices'][i]).any():
            frame_preds.append(None)
        else:
            frame_preds.append({
                'vertices':      torch.from_numpy(data['vertices'][i]),       # (778, 3)
                'cam_t':         data['cam_t'][i],                            # (3,)
                'global_orient': torch.from_numpy(data['global_orient'][i]),  # (1, 3, 3)
                'hand_pose':     torch.from_numpy(data['hand_pose'][i]),      # (15, 3, 3)
                'betas':         torch.from_numpy(data['betas'][i]),          # (10,)
                'is_right':      float(data['is_right'][i]),
                'img_size':      data['img_size'][i],                         # (2,)
                'scaled_focal':  float(data['scaled_focal'][i]),
            })
    return frame_preds, fps


def refine_with_pad_hand(frame_preds, pad_hand_model, mano, device, diff_steps=4, kappa=15/180*np.pi):
    """Run reverse diffusion on sliding windows of length seqlen."""
    seqlen  = 16
    sampler = DiffusionSampler(diff_steps, kappa, device)
    refined = [None] * len(frame_preds)

    for start in tqdm(range(0, len(frame_preds) - seqlen + 1, seqlen), desc='PAD_Hand refinement'):
        seq = frame_preds[start:start + seqlen]
        if any(s is None for s in seq):
            continue

        rotmats   = [torch.cat([fd['global_orient'], fd['hand_pose']], dim=0) for fd in seq]
        rotmats_t = torch.stack(rotmats).to(device)                           # (T, 16, 3, 3)
        betas_t   = torch.stack([fd['betas'] for fd in seq]).to(device)       # (T, 10)

        y0_pose  = rotmat2eulerzxy(rotmats_t.reshape(-1, 3, 3)).reshape(1, seqlen, 48).to(device)
        y0_shape = betas_t.unsqueeze(0).to(device)                            # (1, T, 10)

        y0_shape_t = y0_shape.reshape(-1, 10)
        y0_rotmat  = batch_euler2matzxy(y0_pose.reshape(-1, 3))[0].reshape(-1, 16, 3, 3)
        y0_mano_t  = mano.forward(betas=y0_shape_t, thetas_rotmat=y0_rotmat)
        y0_verts   = y0_mano_t.vertices                                        # (T, 778, 3)

        src_mask     = torch.ones((1, seqlen), dtype=torch.bool, device=device)
        x0_pose_pred = y0_pose.clone()

        with torch.no_grad():
            for n in range(diff_steps, 0, -1):
                if n == diff_steps:
                    n_batch = torch.zeros(1, dtype=torch.float32, device=device) + diff_steps
                    xn_pose = y0_pose + kappa * torch.randn_like(y0_pose).to(device)
                else:
                    n_batch = (torch.zeros(1, device=device) + n).long()
                    xn_pose = sampler.q_sample(x0_pose_pred, y0_pose - x0_pose_pred, n_batch)

                xn_rotmat = batch_euler2matzxy(xn_pose.reshape(-1, 3))[0].reshape(-1, 16, 3, 3)
                xn_mano_t = mano.forward(betas=y0_shape_t, thetas_rotmat=xn_rotmat)

                _, decoder_pred = pad_hand_model.forward(
                    y0_verts.reshape(1, seqlen, 778, 3),
                    xn_mano_t.vertices.reshape(1, seqlen, 778, 3),
                    n_batch.float().to(device),
                    src_mask,
                    mano,
                    y0_shape_t,
                )
                _, _, _, _, x0_rotmat_decoder, x0_mano_decoder = decoder_pred
                x0_pose_pred = rotmat2eulerzxy(x0_rotmat_decoder.reshape(-1, 3, 3)).reshape(1, seqlen, 48)

        refined_verts = x0_mano_decoder.vertices  # (T, 778, 3)
        is_r = seq[0]['is_right']
        for i, gi in enumerate(range(start, start + seqlen)):
            rv = refined_verts[i].detach().cpu().clone()
            rv[:, 0] = (2 * is_r - 1) * rv[:, 0]
            refined[gi] = {**frame_preds[gi], 'refined_vertices': rv}

    for i, fd in enumerate(frame_preds):
        if refined[i] is None and fd is not None:
            refined[i] = {**fd, 'refined_vertices': fd['vertices']}

    return refined


def render_hand(vertices, cam_t, focal, img_size, faces, faces_left, is_right, color=LIGHT_BLUE):
    """Render a hand mesh onto a transparent RGBA canvas using pyrender."""
    w, h = int(img_size[0]), int(img_size[1])
    vert_colors = np.array([(*color, 1.0)] * vertices.shape[0])
    f = faces if is_right else faces_left
    mesh = trimesh.Trimesh(vertices.copy() + cam_t, f.copy(), vertex_colors=vert_colors)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera_pose        = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 0]
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                       cx=w / 2., cy=h / 2., zfar=1e12)
    scene.add(camera, pose=camera_pose)

    # Simple directional lighting
    for phi in [0, 2*np.pi/3, 4*np.pi/3]:
        theta = np.pi / 6
        z = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1., 0., 0.])
        x /= np.linalg.norm(x)
        mat = np.eye(4)
        mat[:3, :3] = np.c_[x, np.cross(z, x), z]
        scene.add_node(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=mat))

    renderer = pyrender.OffscreenRenderer(w, h)
    color_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color_img.astype(np.float32) / 255.0


def put_text_on_image(img, text_up=None, text_down=None, font_size=0.8, font_thickness=2, kwon=False, bg_color=(255, 255, 0)):
    if text_up:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        bg_color = bg_color
        bg_image = np.ones_like(img, dtype=np.uint8) * bg_color
        bg_image = np.ascontiguousarray(bg_image, dtype=np.uint8)
        text_size = cv2.getTextSize(text_up, font, font_size, font_thickness)[0]
        text_x = img.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        bg_image = cv2.putText(bg_image, text_up, (text_x, text_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        img[0:text_y + text_size[1] // 2, text_x - 10:] = bg_image[0:text_y + text_size[1] // 2, text_x - 10:]

    if text_down:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        bg_color = bg_color
        bg_image = np.ones_like(img, dtype=np.uint8) * bg_color
        bg_image = np.ascontiguousarray(bg_image, dtype=np.uint8)
        text_size = cv2.getTextSize(text_down, font, font_size, font_thickness)[0]
        if kwon:
            text_x = img.shape[1] - text_size[0] - 10
            text_y = img.shape[0] - text_size[1] + 10
            bg_image = cv2.putText(bg_image, text_down, (text_x, text_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
            img[text_y - 20:, text_x - 10:] = bg_image[text_y - 20:, text_x - 10:]
        else:
            text_x = img.shape[1] - text_size[0] - 10
            text_y = img.shape[0] - text_size[1] + 15
            bg_image = cv2.putText(bg_image, text_down, (text_x, text_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
            img[text_y - 35:, text_x - 10:] = bg_image[text_y - 35:, text_x - 10:]
    return img


def render_video(refined, video_path, output_path, fps, mano):
    """Save side-by-side video: WiLoR prediction (left) | PAD_Hand-refined (right)."""
    faces      = mano.faces.copy()
    faces_left = mano.faces_left.copy()

    cap = cv2.VideoCapture(video_path)
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frames.append(frame)
    cap.release()

    if not orig_frames:
        return

    h, w = orig_frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

    for fd, orig in tqdm(zip(refined, orig_frames), desc='Rendering', total=len(orig_frames)):
        input_rgb = orig.astype(np.float32)[:, :, ::-1] / 255.0

        if fd is None:
            writer.write(np.hstack([orig, orig]))
            continue

        cam_t  = fd['cam_t']
        is_r   = fd['is_right']
        focal  = fd['scaled_focal']
        imsize = fd['img_size']

        def overlay(verts, color):
            rgba = render_hand(verts, cam_t, focal, imsize, faces, faces_left, is_r, color=color)
            return input_rgb * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]

        def to_bgr(rgb_float):
            return (rgb_float[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

        verts_w = fd['vertices'].numpy() if hasattr(fd['vertices'], 'numpy') else fd['vertices']
        verts_r = fd['refined_vertices'].numpy() if hasattr(fd['refined_vertices'], 'numpy') else fd['refined_vertices']

        # WiLoR label color as BGR uint8 for put_text_on_image background
        blue_bg  = tuple(int(c * 255) for c in LIGHT_BLUE[::-1])
        green_bg = tuple(int(c * 255) for c in LIGHT_GREEN[::-1])

        left_bgr  = to_bgr(overlay(verts_w, LIGHT_BLUE))
        right_bgr = to_bgr(overlay(verts_r, LIGHT_GREEN))

        left_bgr  = put_text_on_image(left_bgr,  text_up='WiLoR', bg_color=blue_bg)
        right_bgr = put_text_on_image(right_bgr, text_up='PAD-Hand',  bg_color=green_bg)

        writer.write(np.hstack([left_bgr, right_bgr]))

    writer.release()


def main():
    parser = argparse.ArgumentParser(description='Demo: WiLoR → PAD_Hand refinement')
    parser.add_argument('--video',               required=True, help='Input video path')
    parser.add_argument('--checkpoint', required=True, help='PAD_Hand checkpoint (.pth)')
    parser.add_argument('--output',         default='demo_output.mp4', help='Output video path')
    parser.add_argument('--diff_steps',     type=int, default=4, help='Diffusion steps')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: WiLoR inference (runs in wilor conda env via subprocess)
    npz_path    = run_wilor_inference(args.video)
    frame_preds, fps = load_wilor_results(npz_path)
    fps = 8
    os.remove(npz_path)

    n_det = sum(1 for f in frame_preds if f is not None)
    print(f'{len(frame_preds)} frames, {n_det} with detections')

    # Step 2: PAD_Hand refinement (runs in current env)
    print('Loading PAD_Hand...')
    pad_hand_model = load_pad_hand_model(args.checkpoint, device)
    mano = MANO('RIGHT', device)

    print('Refining with PAD_Hand...')
    refined = refine_with_pad_hand(frame_preds, pad_hand_model, mano, device, args.diff_steps)

    # Step 3: Render
    print('Rendering output video...')
    render_video(refined, args.video, args.output, fps, mano)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
