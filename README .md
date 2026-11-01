# No-GPU TRYON: Virtual Fashion Assistant

A virtual try-on and fashion assistant that runs **without a GPU**. Given a photo of a person and a garment, it parses the body, warps the garment onto the person, and adds assistant features on top: counterfactual edits, physics-based plausibility checks, and uncertainty estimation.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Assistant Modules](#assistant-modules)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **CPU-friendly pipeline**: runs without a GPU (see `no_Gpu_version.ipynb` for a ready-to-run notebook).
- **Human parsing**: DeepLab-Xception models with graph convolutions over label graphs (`gcn.py`, `graph.py`).
- **Garment warping**: geometric matching module (`networks_gmm.py`) that warps the cloth to the person's pose.
- **Pose-aware**: reads OpenPose JSON keypoints (`openpose_json.py`).
- **Counterfactual module**: "what if" edits such as body scaling and pose changes.
- **Physics module**: fabric strain, pose constraints and a violation score for fit plausibility.
- **Uncertainty module**: MC dropout, confidence scores and uncertainty maps.
- **Visualization**: result rendering through `visualization.py`.

---

## How It Works

```
Person image + Garment image
        │
        ▼
 Preprocessing  (cloth_mask.py, image_mask.py, custom_transforms.py)
        │
        ▼
 Human parsing  (DeepLab-Xception + graph convolutions)
        │
        ▼
 Pose keypoints (openpose_json.py)
        │
        ▼
 Garment warping (networks_gmm.py) ──► warp-cloth / warp-mask
        │
        ├──► Counterfactual edits   (counterfactual_module/)
        ├──► Physics checks         (physics_module/)
        ├──► Uncertainty scoring    (uncertainty_module/)
        ▼
 Visualization (visualization.py)
```

`inference.py` is the main entry point for running a user image through the models, and `test.py` runs the try-on test pass over the pairs listed in `data/test_pairs.txt`.

---

## Repository Structure

```
No-gPU-TRYON-main/
├── counterfactual_module/
│   ├── body_scaler.py          # Scale body shape/size for what-if edits
│   ├── cf_generator.py         # Generates counterfactual results
│   └── pose_editor.py          # Edit pose keypoints
├── data/
│   ├── test/
│   │   ├── cloth/              # Garment images
│   │   ├── cloth-mask/         # Garment masks
│   │   ├── image/              # Person images
│   │   ├── image-mask/         # Person masks
│   │   ├── image-parse/        # Human parsing maps
│   │   ├── image-parse-new/    # Updated parsing maps
│   │   ├── pose/               # OpenPose keypoint JSON files
│   │   ├── warp-cloth/         # Output: warped garments
│   │   └── warp-mask/          # Output: warped garment masks
│   ├── grid.png                # Grid used by the warping module
│   └── test_pairs.txt          # Person / garment test pairs
├── dataloaders/
│   └── custom_transforms.py    # Image transforms
├── networks/
│   ├── __init__.py
│   ├── deeplab_xception.py            # DeepLab with Xception backbone
│   ├── deeplab_xception_synBN.py      # SynBN variant
│   ├── deeplab_xception_transfer.py   # Transfer segmentation model
│   ├── deeplab_xception_universal.py  # Universal transfer model
│   ├── gcn.py                         # Graph convolution layers
│   └── graph.py                       # Label graph definitions
├── physics_module/
│   ├── fabric_strain.py        # Fabric stretch/strain estimation
│   ├── pose_constraints.py     # Pose-based fit constraints
│   └── violation_score.py      # Plausibility violation score
├── sync_batchnorm/             # Synchronized batch norm
│   ├── batchnorm.py
│   ├── comm.py
│   └── replicate.py
├── uncertainty_module/
│   ├── confidence_score.py     # Prediction confidence
│   ├── mc_dropout.py           # Monte Carlo dropout
│   └── uncertainty_map.py      # Per-pixel uncertainty maps
├── cloth_mask.py               # Garment mask generation
├── cp_dataset.py               # Person/cloth dataset loader
├── image_mask.py               # Person mask generation
├── inference.py                # Inference entry point
├── networks_gmm.py             # Geometric matching (garment warping)
├── no_Gpu_version.ipynb        # CPU-only notebook version
├── openpose_json.py            # OpenPose JSON parsing
├── test.py                     # Try-on test script
├── visualization.py            # Result visualization
└── README.md
```

---

## Installation

The compiled files in the repo were built with **Python 3.7**, so use 3.7 or a close version.

```bash
git clone https://github.com/<your-username>/No-gPU-TRYON.git
cd No-gPU-TRYON

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Typical dependencies: PyTorch (CPU build is enough), torchvision, NumPy, OpenCV, Pillow, matplotlib, tqdm.

---

## Data Preparation

Put your inputs under `data/test/`, using the same file name for each person image and its related files:

| Folder | Contents |
|---|---|
| `image/` | Person photos |
| `image-parse/` | Human parsing maps |
| `image-mask/` | Person masks (see `image_mask.py`) |
| `cloth/` | Garment images |
| `cloth-mask/` | Garment masks (see `cloth_mask.py`) |
| `pose/` | OpenPose JSON keypoints |

List the person and garment pairs to try on in `data/test_pairs.txt`, one pair per line:

```
person_001.jpg cloth_001.jpg
person_002.jpg cloth_002.jpg
```

---

## Usage

### Run the try-on test

```bash
python test.py
```

Warped garments and masks are saved to `data/test/warp-cloth/` and `data/test/warp-mask/`.

### Run inference on a single image

```bash
python inference.py --image path/to/person.jpg
```

### Run the CPU notebook

Open `no_Gpu_version.ipynb` in Jupyter or Google Colab and run the cells in order.

> Update the commands above to match the actual arguments your scripts accept.

---

## Assistant Modules

### Counterfactual module
Explore "what if" scenarios without redoing the whole pipeline.
- `body_scaler.py`: change body proportions.
- `pose_editor.py`: modify pose keypoints.
- `cf_generator.py`: produce the counterfactual try-on result.

### Physics module
Checks whether the garment fit is physically plausible.
- `fabric_strain.py`: how much the fabric would stretch.
- `pose_constraints.py`: whether the pose allows the fit.
- `violation_score.py`: a single score summarising the violations.

### Uncertainty module
Shows how much to trust the result.
- `mc_dropout.py`: repeated stochastic passes for uncertainty estimates.
- `confidence_score.py`: confidence of the prediction.
- `uncertainty_map.py`: visual map of uncertain regions.

---


