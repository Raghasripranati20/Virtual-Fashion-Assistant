# Virtual Fashion Assistant

A virtual fashion assistant that takes a photo of a person, understands their body and clothing through human parsing and segmentation, and supports virtual try-on of garments. The project combines a graph-based human parsing model, a DeepLab segmentation backbone, a garment-warping try-on pipeline, and a set of assistant extensions for pose, physics, uncertainty and visualization.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Assistant Extensions](#assistant-extensions)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Image inference**: submit a user image and get a segmentation / parsing result.
- **Human parsing**: transfer segmentation with graph convolutions over label graphs.
- **Strong backbone**: DeepLab with an Xception backbone and a SynBN (synchronized batch norm) variant.
- **Multi-GPU training**: parallel replication and synchronized batch normalization across devices.
- **Virtual try-on**: cloth masks, image masks and geometric-matching garment warping.
- **Assistant extensions**: pose keypoints, physics checks, counterfactual edits, uncertainty scoring and result visualization.

---

## Architecture

The system is organised into five groups.

### 1. Inference workflow
| Component | File | Role |
|---|---|---|
| Image inference | `inference.py` | Entry point. Receives the user's image, preprocesses it and invokes the model. |
| Image transforms | – | Resizing, normalization and tensor conversion applied before inference. |

### 2. Vision models
| Component | File | Role |
|---|---|---|
| Transfer segmentation | – | Main segmentation model invoked by inference. |
| Universal transfer model | – | Generalised transfer model that shares the same building blocks. |
| Graph convolutions | `gcn.py` | Graph layers used by both transfer models. |
| Label graphs | `graph.py` | Graph definitions (label relations) used by the graph convolutions. |
| DeepLab Xception | – | Backbone inherited by both transfer models. |
| SynBN DeepLab | – | DeepLab variant using synchronized batch norm. |

### 3. Parallel computation
| Component | File | Role |
|---|---|---|
| Parallel replication | `replicate.py` | Replicates modules across GPUs and registers replicas. |
| Synchronized batch norm | `batchnorm.py` | Batch norm whose statistics are shared across devices. |
| Sync communication | `comm.py` | Communication primitive that coordinates statistics. |

### 4. Try-on and data
| Component | File | Role |
|---|---|---|
| Cloth masks | `cloth_mask.py` | Generates garment masks. |
| Garment warping | `networks_gmm.py` | Geometric matching module that warps garments to the body. |
| Try-on dataset | `cp_dataset.py` | Dataset loader for person / cloth pairs. |
| Image masks | `image_mask.py` | Generates person image masks. |

### 5. Assistant extensions
| Component | File | Role |
|---|---|---|
| Pose keypoints | `openpose_json.py` | Reads OpenPose JSON keypoints. |
| Result visualization | `visualization.py` | Renders segmentation / try-on outputs. |
| Physics checks | – | Plausibility checks on garment fit and drape. |
| Counterfactual edits | – | "What if" edits, such as a different garment or colour. |
| Uncertainty scoring | – | Confidence estimates for predictions. |

### Data flow

```
User image
   │  submits image
   ▼
inference.py ──preprocesses with──► Image transforms
   │  invokes model
   ▼
Transfer segmentation ──► Graph convolutions ──► Label graphs
   │                  └─► DeepLab Xception
   └─► SynBN DeepLab ──► Synchronized batch norm ──► Sync communication
                                  ▲
                      Parallel replication
```

---

## Project Structure

```
virtual-fashion-assistant/
├── inference.py          # Image inference entry point
├── gcn.py                # Graph convolution layers
├── graph.py              # Label graph definitions
├── networks_gmm.py       # Garment warping (GMM)
├── cp_dataset.py         # Try-on dataset
├── cloth_mask.py         # Cloth mask generation
├── image_mask.py         # Image mask generation
├── openpose_json.py      # Pose keypoint parsing
├── visualization.py      # Result visualization
└── sync_batchnorm/
    ├── batchnorm.py      # Synchronized batch norm
    ├── comm.py           # Cross-device communication
    └── replicate.py      # Parallel replication
```

> Adjust the paths above to match your repository layout.

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements** (typical): Python 3.8+, PyTorch, torchvision, NumPy, OpenCV, Pillow, matplotlib. A CUDA GPU is recommended, and is needed for multi-GPU synchronized batch norm.

---

## Usage

### Run inference on an image

```bash
python inference.py --image path/to/person.jpg --output results/
```

### Prepare try-on inputs

```bash
python cloth_mask.py   --input data/cloth --output data/cloth-mask
python image_mask.py   --input data/image --output data/image-mask
```

### Visualize results

```bash
python visualization.py --input results/ --output results/vis/
```

> Replace the flags above with the actual arguments your scripts accept.

---

## Datasets

The try-on pipeline uses person / garment pairs loaded by `cp_dataset.py`. Typical inputs per sample:

- Person image
- Garment image and its mask
- Body parsing map
- Pose keypoints (OpenPose JSON)

Place the data under `data/` and update the paths in your config.

---

## Assistant Extensions

These modules build on the core pipeline to make the assistant more useful and trustworthy:

- **Pose keypoints**: body pose from OpenPose JSON to guide warping and fitting.
- **Physics checks**: validate that garments fit and drape plausibly.
- **Counterfactual edits**: preview alternatives, such as changing garment or colour, without re-running the full pipeline.
- **Uncertainty scoring**: flag low-confidence regions or predictions.
- **Result visualization**: overlays, segmentation maps and try-on comparisons.

---



