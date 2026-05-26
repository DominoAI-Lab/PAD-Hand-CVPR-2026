<h1 align="center">PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery</h1>

<p align="center">
  <a href="https://cvpr.thecvf.com/Conferences/2026">
    <img src="https://img.shields.io/badge/CVPR-2026-blue?style=for-the-badge" alt="CVPR 2026"/>
  </a>
  &nbsp;
  <a href="https://cvpr.thecvf.com/Conferences/2026">
    <img src="https://img.shields.io/badge/%E2%AD%90%20Highlight%20Paper-%23FFD700?style=for-the-badge" alt="Highlight Paper"/>
  </a>
</p>

<p align="center">
  <a href="https://elkhanzada.github.io/pad-hand/">
    <img src="https://img.shields.io/badge/Project%20Page-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"/>
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2603.26068">
    <img src="https://img.shields.io/badge/📄%20arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
</p>

<p align="center">
  <b>Elkhan Ismayilzada</b><sup>1</sup> &nbsp;&nbsp;
  <b>Yufei Zhang</b><sup>2</sup> &nbsp;&nbsp;
  <b>Zijun Cui</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Michigan State University &nbsp;&nbsp;
  <sup>2</sup>Independent Researcher
</p>

---

## News
- [x] `2026-05-26`: Demo code and pretrained checkpoint released.
- [x] `2026-04-09`: PAD-Hand selected as a **Highlight** at CVPR 2026!
- [x] `2026-02-20`: PAD-Hand accepted at CVPR 2026!

---

## Setup

This project uses **two separate conda environments**:

| Environment | Purpose |
|-------------|---------|
| `wilor` | WiLoR hand detection and initial pose estimation |
| `pad_hand` | PAD-Hand diffusion refinement and rendering |

### 1. WiLoR Environment (`wilor`)

Follow the installation instructions in the [WiLoR repository](https://github.com/rolpotamias/WiLoR).

Then initialize the submodule:

```bash
git submodule update --init --recursive
```

Place pretrained WiLoR weights under:

```
WiLoR/pretrained_models/
├── wilor_final.ckpt
├── detector.pt
├── model_config.yaml
└── dataset_config.yaml
```

### 2. PAD-Hand Environment (`pad_hand`)

```bash
conda create -n pad_hand python=3.7
conda activate pad_hand
pip install -r requirements.txt
```

> **Note:** `torch-scatter` requires matching your CUDA and PyTorch versions. See the [torch-scatter installation guide](https://github.com/rusty1s/pytorch_scatter).

### 3. Assets

The MANO model files must be downloaded manually due to licensing. Please register and download from the [MANO website](https://mano.is.tue.mpg.de/) and place the processed files under:

```
assets/
├── MANO_RIGHT.pkl
├── MANO_LEFT.pkl
├── MeshConv_template.ply
└── MeshConv_transform.pkl
```

Then run the preprocessing script to convert the official MANO files into the format expected by this codebase:

```bash
conda activate pad_hand
python mano_preprocessing.py
```

The mesh convolution module is adapted from [MobRecon](https://github.com/SeanChenxy/HandMesh). The mesh data processing module can be installed from [psbody-mesh](https://github.com/MPI-IS/mesh). We thank them for generously sharing their outstanding work.

---

## Demo

```bash
conda activate pad_hand
python demo.py \
  --video path/to/input.mp4 \
  --checkpoint path/to/pad_hand.pth \
  --output output.mp4
```

**Example:**

```bash
python demo.py --video demo_input.mp4 --checkpoint pad_hand.pt
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Input video path |
| `--checkpoint` | Yes | — | PAD-Hand model checkpoint (`.pth`) — [download here](https://drive.google.com/file/d/1juZR9XxhQiQSWFgcXJOITxQ386Tkr8S5/) |
| `--output` | No | `demo_output.mp4` | Output video path |

The output is a side-by-side video: **WiLoR prediction** (left, blue) vs **PAD-Hand refined** (right, green).

> WiLoR inference is automatically invoked as a subprocess inside `demo.py` using `conda run -n wilor`. No manual step needed.

---

## Acknowledgements
This repository is based on
* [DIP-Hand](https://github.com/zhangy76/DIP-Hand/)
* [BayesDiff](https://github.com/karrykkk/BayesDiff)

## Citation

```bibtex
@InProceedings{Ismayilzada_2026_CVPR,
    author    = {Ismayilzada, Elkhan and Zhang, Yufei and Cui, Zijun},
    title     = {PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {28358-28368}
}
```
