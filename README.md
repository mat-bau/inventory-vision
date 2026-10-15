# InventoryVision
A computer vision system for detecting, classifying, and analyzing electronic components on PCB boards using YOLOv11.

## Features
- **Component Detection**: Detects 8 types of electronic components (resistors, inductors, capacitors, ICs, diodes, LEDs, connectors, transistors)
- **Classification**: Classifies detected components with confidence scores
- **Mass Estimation**: Estimates component mass based on visual area using statistical models
- **Human Correction**: Interactive tool for validating and correcting detections
- **Data Export**: Generates annotated images and detection reports
- **Model Training**: Fine-tunes a YOLOv11 model on a custom PCB component dataset, with automated dataset splitting, reproducible seeding, and per-split evaluation

## Installation
Requires Python 3.13 and [uv](https://docs.astral.sh/uv/)
```bash
uv sync
```

## Usage

### Inference
```bash
uv run python main.py --input <image_path> --model <model_path> [options]
```
**Required arguments:**
- `--input` - Path(s) to input image(s) (supports multiple files)
- `--model` - Path to the YOLO model (.pt file)

**Optional arguments:**
- `--output-xlsx` - Output Excel file path
- `--debug` - Save debug images with bounding boxes
- `--human` - Enable human-in-the-loop correction (interactive GUI)
- `--metrics` - Display detailed component dimension metrics
- `--bbox-reduction` - Shrink bboxes by percentage (e.g., 4)
- `--bbox-uncertainty` - Uncertainty on bbox dimensions in %
- `--individual-ic` - Output individual IC areas to separate file
- `--taxonomy` - Subcategory definitions (default `config/taxonomy.yaml`)
- `--impact-factors` - Impact factors per ecoinvent flow (default `config/impact_factors.yaml`)
- `--gui-size` - Size of the review window, e.g. `1360x820` (default)

**Example:**
```bash
uv run python main.py --input board.jpg --model model/best.pt --debug --human --metrics
```

### Human review (`--human`)
One window: image on the left, control panel on the right. Everything works with
the left button and the keyboard. Press **H** in the
window for the full list of controls.

| Action | How |
|---|---|
| Select a box | MODIFY mode (**M**): click it. Click again at the same spot to cycle through overlapping boxes. |
| Change class | Click a class in the panel, or keys **1-9** |
| Set subcategory | Click a subcategory in the panel, or **[ ]** (**0** = unspecified) |
| Move / resize | Drag the selected box / drag one of its white handles |
| Add a box | ADD mode (**A**): drag. The new box is selected: pick its class and subcategory right away. |
| Delete | DELETE mode (**D**) and click, or **X** / Delete / Backspace on the selection |
| Zoom / pan | Wheel or **+ -**, **F** to fit; drag the background (Shift-drag or right-drag in ADD mode), arrow keys |
| Review | **N / P**: next / previous detection, lowest confidence first |
| Undo / redo | **Z** or Ctrl-Z / **Y** or Ctrl-Y |
| Save / quit | **S** / **Q** (every change is also saved automatically) |

Platform notes: on macOS the two-finger vertical scroll zooms and the horizontal
scroll pans.
On Linux (OpenCV's Qt backend) Qt zooms the window by itself on every wheel
event; the GUI cancels this by redrawing the window, so a short flicker while
scrolling is expected (**+ / -** zoom without it).

The review state is stored in `output/labels/<image>_state.json` and reloaded on
the next run. Corrections are counted from that state: additions, deletions, reclassifications (the
operations counted in the paper), plus moves/resizes and subcategories.

### Subcategories and impacts
For now, the automated detector only predicts the 8 classes (+ `Switch`, present in `model/best.pt`).
Subcategories are chosen by the human after inference and drive the LCI:

- `config/taxonomy.yaml` lists, per class, its subcategories with the ecoinvent
  reference product to use and, for ICs, the die-to-package ratio (DPR) of the
  package type (Teehan 2014 for unknown packages, Proske et al. 2024 per package).
  `unspecified` reproduces the original behaviour exactly. Flows marked
  `flow_verified: false` must be checked against your ecoinvent version.
- `config/impact_factors.yaml` ecoinvent factors and the die-area -> kg conversion of the IC datasets from Teehan 2014. Impacts are only computed for
  flows that have factors; the others are listed as missing.

The subcategory is stored as a 7th column of the label files
(`cls xc yc w h conf subcategory`, omitted when unspecified). The Excel output
keeps the original `Component Metrics` sheet and adds:

| Sheet | Content |
|---|---|
| `LCI by subcategory` | count, area, mass (mg) or die area (mm²) with bounds, reference quantity, impacts |
| `Components` | one row per detection (image, class, subcategory, size, mass or DPR / die area) |
| `Impacts` | totals per impact category, coverage, missing factors |
| `Assumptions` | DPR and flow used for every subcategory, sources, uncertainty conventions |

Uncertainty conventions: mass bounds combine the regression prediction interval
and the bbox uncertainty in quadrature (as in `component_to_mass.py`); die-area
bounds are scenarios (all ICs at the low or high end of their DPR range); impact
totals add the row bounds linearly.

### Tests
```bash
uv run --group dev pytest
```
The tests do not need a GPU nor the model: the GUI is driven headlessly and the
detector is stubbed in the end-to-end test.

### Training
```bash
uv run python scripts/train.py [options]
```
Fine-tunes a YOLOv11 model (`yolo11m` by default) on a PCB component dataset: splits the data into train/valid/test, trains with a preconfigured schedule, and evaluates the result on each split.

**Optional arguments:**
- `--dataset` - Path to the source dataset directory, containing `images/`, `labels/`, and `data.yaml` (defaults to `dataset/real_dataset`)
- `--pretrained-model` - Path to pretrained/base model weights to fine-tune from (defaults to `weights/yolo11m.pt`)
- `--cpu` - Train on CPU instead of GPU
- `--output-name` - Custom name for the run's output directory (defaults to `output_yolo11m_seed_<seed>_batch_<batch>`)
- `--seed` - Random seed for reproducibility (default: `14`)

**Expected dataset layout** (source directory passed via `--dataset`):
```
dataset/real_dataset/
├── data.yaml          # or data.yml — must define nc and names
├── images/            # all images, e.g. board_<id>_<n>.png
└── labels/            # YOLO-format .txt labels, one per image
```

## Project Structure
- `main.py` - Main entry point with interactive CLI for detection/inference
- `scripts/` - Core processing modules:
  - `detection_classification.py` - YOLO detection and annotation
  - `crop_board.py` - Board region extraction
  - `component_to_mass.py` - Mass estimation models
  - `transform_detections.py` - Detection format conversion
  - `human_correction.py` - Manual review interface (classes, subcategories, boxes)
  - `taxonomy.py` - Loads the class/subcategory taxonomy
  - `lci.py` - Subcategory-level inventory (mass, die area via DPR) and impacts
  - `labels.py` - Label files with the optional subcategory column
  - `train.py` - Training entry point: dataset preparation, YOLOv11 fine-tuning, and evaluation
- `config/taxonomy.yaml` - Subcategories, ecoinvent flows, DPR per IC package type
- `config/impact_factors.yaml` - Impact factor template
- `tests/` - Unit, GUI (headless) and end-to-end tests
- `model/best.pt` - Pre-trained YOLOv11 finetuned model
- `weights/yolo11m.pt` - Default base YOLOv11 weights used as the starting point for fine-tuning
- `dataset/real_dataset` - Default source dataset used by `train.py`

## Dependencies
- OpenCV, PyTorch, Ultralytics (YOLO), Pandas, Rich CLI, PyYAML, NumPy
