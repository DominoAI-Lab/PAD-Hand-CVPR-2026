"""Run WiLoR on a video and save per-frame predictions to a .npz file.
Called automatically by demo.py via: conda run -n wilor python wilor_inference.py
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
_WILOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WiLoR')
sys.path.insert(0, _WILOR_DIR)

import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

_DEMO_DIR   = os.path.dirname(os.path.abspath(__file__))
WILOR_CKPT  = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/wilor_final.ckpt')
WILOR_CFG   = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/model_config.yaml')
DETECTOR_PT = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/detector.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  required=True)
    parser.add_argument('--output', required=True, help='Path to save .npz results')
    args = parser.parse_args()
    # Resolve to absolute paths before chdir so relative paths keep working
    args.video  = os.path.abspath(args.video)
    args.output = os.path.abspath(args.output)
    os.chdir(_WILOR_DIR)  # WiLoR expects cwd to be its own dir (mano_data/, pretrained_models/)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wilor_model, wilor_cfg = load_wilor(checkpoint_path=WILOR_CKPT, cfg_path=WILOR_CFG)
    detector = YOLO(DETECTOR_PT)
    wilor_model = wilor_model.to(device)
    detector    = detector.to(device)
    wilor_model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Per-frame arrays; NaN-filled rows mean no detection
    all_vertices      = []  # (778, 3) or None
    all_cam_t         = []  # (3,)     or None
    all_global_orient = []  # (1,3,3)  or None
    all_hand_pose     = []  # (15,3,3) or None
    all_betas         = []  # (10,)    or None
    all_is_right      = []  # float    or None
    all_img_size      = []  # (2,)     or None
    all_scaled_focal  = []  # float    or None

    pbar = tqdm(desc='WiLoR inference')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        detections = detector(frame, conf=0.3, verbose=False)[0]
        bboxes, rights = [], []
        for det in detections:
            bbox_data = det.boxes.data.cpu().detach().squeeze()
            if bbox_data.dim() == 0:
                continue
            rights.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
            bboxes.append(bbox_data[:4].tolist())

        if not bboxes:
            for lst in (all_vertices, all_cam_t, all_global_orient, all_hand_pose,
                        all_betas, all_is_right, all_img_size, all_scaled_focal):
                lst.append(None)
            continue

        dataset    = ViTDetDataset(wilor_cfg, frame, np.stack(bboxes), np.stack(rights))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        frame_hands = []
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = wilor_model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam'].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            img_size     = batch['img_size'].float()
            scaled_focal = wilor_cfg.EXTRA.FOCAL_LENGTH / wilor_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t   = cam_crop_to_full(
                pred_cam, batch['box_center'].float(), batch['box_size'].float(),
                img_size, scaled_focal,
            ).detach().cpu()

            for n in range(batch['img'].shape[0]):
                is_r  = float(batch['right'][n].cpu().item())
                verts = out['pred_vertices'][n].detach().cpu().clone()
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]

                frame_hands.append({
                    'vertices':      verts.numpy(),
                    'cam_t':         pred_cam_t[n].numpy(),
                    'global_orient': out['pred_mano_params']['global_orient'][n].detach().cpu().numpy(),
                    'hand_pose':     out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy(),
                    'betas':         out['pred_mano_params']['betas'][n].detach().cpu().numpy(),
                    'is_right':      is_r,
                    'img_size':      img_size[n].cpu().numpy(),
                    'scaled_focal':  float(scaled_focal.cpu()),
                })

        chosen = next((h for h in frame_hands if h['is_right'] > 0.5), None)
        if chosen is None and frame_hands:
            chosen = frame_hands[0]

        if chosen is None:
            for lst in (all_vertices, all_cam_t, all_global_orient, all_hand_pose,
                        all_betas, all_is_right, all_img_size, all_scaled_focal):
                lst.append(None)
        else:
            all_vertices.append(chosen['vertices'])
            all_cam_t.append(chosen['cam_t'])
            all_global_orient.append(chosen['global_orient'])
            all_hand_pose.append(chosen['hand_pose'])
            all_betas.append(chosen['betas'])
            all_is_right.append(chosen['is_right'])
            all_img_size.append(chosen['img_size'])
            all_scaled_focal.append(chosen['scaled_focal'])

    cap.release()
    pbar.close()

    n = len(all_vertices)
    # Use NaN sentinels for missing frames so we can store as dense arrays
    def stack_or_nan(lst, shape):
        out = np.full((n, *shape), np.nan, dtype=np.float32)
        for i, v in enumerate(lst):
            if v is not None:
                out[i] = v
        return out

    np.savez(args.output,
        vertices      = stack_or_nan(all_vertices,      (778, 3)),
        cam_t         = stack_or_nan(all_cam_t,         (3,)),
        global_orient = stack_or_nan(all_global_orient, (1, 3, 3)),
        hand_pose     = stack_or_nan(all_hand_pose,     (15, 3, 3)),
        betas         = stack_or_nan(all_betas,         (10,)),
        is_right      = np.array([v if v is not None else np.nan for v in all_is_right], dtype=np.float32),
        img_size      = stack_or_nan(all_img_size,      (2,)),
        scaled_focal  = np.array([v if v is not None else np.nan for v in all_scaled_focal], dtype=np.float32),
        fps           = np.array([fps], dtype=np.float32),
    )
    print(f'Saved WiLoR predictions for {n} frames → {args.output}')


if __name__ == '__main__':
    main()
