from pathlib import Path

import cv2
from ultralytics import YOLO


def save_annotated_image(img, detections, out_path: Path):
    """
    Save an annotated copy of the image with bounding boxes and labels.

    Args:
        img: Image to annotate
        detections: List of detections
        out_path: Path to save annotated image
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for det in detections:
        b = det['bbox']
        x1, y1, x2, y2 = map(int, (b['x1'], b['y1'], b['x2'], b['y2']))
        color = (0, 255, 0)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def create_output_dirs(base_dir: str):
    """
    Create output directories for annotated images and label files.

    Args:
        base_dir: Base directory for output directories
    """
    base = Path(base_dir)
    ann_dir = base / "annotated_images"
    lbl_dir = base / "labels"
    ann_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir, lbl_dir

def shrink_bbox(bbox, shrink_percent, img_width, img_height):
    """
    Shrink a bounding box around its center.

    Args:
        bbox: dict {x1,y1,x2,y2}
        shrink_percent: float (e.g. 4.0 = shrink by 4%)
        img_width, img_height: image size for clamping
    """
    if shrink_percent <= 0.0:
        return bbox

    scale = 1.0 - shrink_percent / 100.0

    cx = (bbox['x1'] + bbox['x2']) / 2
    cy = (bbox['y1'] + bbox['y2']) / 2
    w = (bbox['x2'] - bbox['x1']) * scale
    h = (bbox['y2'] - bbox['y1']) * scale

    x1 = max(0, cx - w / 2)
    y1 = max(0, cy - h / 2)
    x2 = min(img_width - 1, cx + w / 2)
    y2 = min(img_height - 1, cy + h / 2)

    return {
        'x1': float(x1),
        'y1': float(y1),
        'x2': float(x2),
        'y2': float(y2),
    }

class PCBDetector:

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    """
    Defined image extensions
    """

    def __init__(self, model_path: str, bbox_reduction: float = 0.0):
        """
        YOLO model for PCB component detection.

        Args:
            model_path: Path to YOLO model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        print("Using class names embedded in the model:")
        print(self.class_names)
        self.bbox_reduction = bbox_reduction

    # ------------------------
    #   Internal utilities
    # ------------------------

    def save_yolo_txt(self, img_shape, img_name, detections, lbl_dir: Path):
        """
        Save detections in YOLO .txt format.

        Args:
            img_shape: Image shape
            img_name: Image name
            detections: List of detections
            lbl_dir: Path to save annotated image
        """
        height, width = img_shape[:2]
        lines = []
        for det in detections:
            cls_id = next(i for i, n in self.class_names.items() if n == det['class'])
            b = det['bbox']
            x_center = ((b['x1'] + b['x2']) / 2) / width
            y_center = ((b['y1'] + b['y2']) / 2) / height
            box_w = (b['x2'] - b['x1']) / width
            box_h = (b['y2'] - b['y1']) / height
            # confidence kept as 6th column: shown in the review GUI and used for its review order
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} {det['confidence']:.6f}")

        out_txt = lbl_dir / f"{img_name}.txt"
        out_txt.write_text("\n".join(lines))

    def infer_image(self, img_path: Path):
        """
        Run YOLO inference on a single image and return the raw result.

        Args:
            img_path: Path to image

        Return:
            Model inference
        """
        return self.model.predict(
            source=str(img_path),
            conf=0.0,  # no threshold, filtering is applied later
            save=False,
            verbose=False
        )[0]

    def process_image(self, img_path: Path, conf_thresh, ann_dir: Path, lbl_dir: Path, annotate: bool):
        """
        Run inference, filter results, and save detections for one image.

        Args:
            img_path: Path to image
            conf_thresh: Confidence threshold
            ann_dir: Path to annotated image
            lbl_dir: Path to save annotated image
            annotate: Whether to annotate or not

        Return:
            Model predictions
        """
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: failed to load {img_path}")
            return []

        result = self.infer_image(img_path)
        if not result.boxes or len(result.boxes) == 0:
            print(f"No boxes detected for {img_path.name}")
            return []

        # Extract boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        detections = []
        for idx, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
            class_name = self.class_names[int(cl)]
            threshold = conf_thresh.get(class_name, 0.25) if isinstance(conf_thresh, dict) else conf_thresh
            if score < threshold:
                continue
            img_h, img_w = img.shape[:2]
            bbox = {
                'x1': float(box[0]), 'y1': float(box[1]),
                'x2': float(box[2]), 'y2': float(box[3])
            }
            bbox = shrink_bbox(
                bbox,
                self.bbox_reduction,
                img_w,
                img_h
            )
            detections.append({
                'id': idx,
                'class': class_name,
                'confidence': float(score),
                'bbox': bbox
            })

        # Save outputs
        self.save_yolo_txt(img.shape, img_path.stem, detections, lbl_dir)
        if annotate:
            out_img = ann_dir / f"{img_path.stem}_annotated.jpg"
            save_annotated_image(img, detections, out_img)

        return detections


    def run(self, source: str, output_dir: str, conf_thresh=0.25, annotate=True):
        """
        Run detection on a single image or on all images in a directory.

        Args:
            source: Path to image or directory
            output_dir: Path to output directory
            conf_thresh: Confidence threshold
            annotate: Whether to annotate or not
        """
        ann_dir, lbl_dir = create_output_dirs(output_dir)
        src = Path(source)

        # Collect images
        if src.is_file() and src.suffix.lower() in self.IMAGE_EXTS:
            images = [src]
        elif src.is_dir():
            images = [p for p in src.glob("*") if p.suffix.lower() in self.IMAGE_EXTS]
        else:
            raise ValueError(f"Invalid source path: {source}")

        if not images:
            print(f"No images found in {source}")
            return

        for img in images:
            print(f"Processing {img.name}")
            self.process_image(img, conf_thresh, ann_dir, lbl_dir, annotate)

        print(f"\nOutputs saved in: {output_dir}")