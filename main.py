import argparse
import json
import os
import sys

import cv2
import pandas as pd
import pyfiglet
from rich import print
from ultralytics import YOLO

from scripts import component_to_mass, crop_board, detection_classification, lci
from scripts.labels import read_labels, write_labels
from scripts.taxonomy import UNSPECIFIED, Taxonomy
from scripts.transform_detections import transform_yolo_detections


def ask_positive_float(prompt):
    """
    Prompt the user to enter a positive floating-point number.
    Repeats until a valid positive number is entered.

    Args:
        prompt (str): The prompt to be displayed.

    Return:
        floating-point number: The user entered a positive floating-point number.
    """
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a valid positive number.")


def analyze_detection_file(detection_file, class_map, conf_thresholds, title="DETECTIONS"):
    """
    Analyze a YOLO detection file and print statistics
    ONLY for detections that meet or exceed confidence thresholds
    
    Returns:
        dict: class_counts {class_name: count} - ONLY above threshold
        list: low_conf_detections [(class_name, confidence)] - below threshold (for warning)
    """
    if not os.path.exists(detection_file):
        print(f"⚠️ Warning: Detection file not found: {detection_file}")
        return {}, []
    
    if os.path.getsize(detection_file) == 0:
        print("⚠️ Warning: Detection file is empty")
        return {}, []
    
    class_counts = {}  # Only counts above threshold
    low_conf_detections = []  # Track below threshold
    total_detections = 0
    total_above_threshold = 0

    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                total_detections += 1
                cls_id = int(parts[0])
                cls_name = class_map.get(cls_id, f"unknown_{cls_id}")
                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                
                # Check confidence threshold
                threshold = conf_thresholds.get(cls_name, 0.5)
                
                if confidence >= threshold:
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_above_threshold += 1
                else:
                    low_conf_detections.append((cls_name, confidence))

    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    print(f"Raw detections in file:       {total_detections}")
    print(f"Detections >= threshold:      {total_above_threshold}")
    print(f"Detections < threshold:       {len(low_conf_detections)}")
    print()
    
    if class_counts:
        for cls_name, count in sorted(class_counts.items()):
            threshold = conf_thresholds.get(cls_name, 0.5)
            print(f"  ✓ {cls_name:15s}: {count:3d} detections (threshold: {threshold:.2f})")
    else:
        print("  No detections above threshold")

    if low_conf_detections:
        print(f"\n⚠️  {len(low_conf_detections)} rejected detections (< threshold) :")
        rejected_by_class = {}
        for cls_name, conf in low_conf_detections:
            rejected_by_class[cls_name] = rejected_by_class.get(cls_name, 0) + 1
        
        for cls_name, count in sorted(rejected_by_class.items()):
            threshold = conf_thresholds.get(cls_name, 0.5)
            print(f"     • {cls_name:15s}: {count:3d} rejected (threshold: {threshold:.2f})")

    print("="*60 + "\n")
    
    return class_counts, low_conf_detections


def extract_individual_ic_areas(labels_file, class_map, conf_thresholds, image_width_mm, image_height_mm, image_label):
    """
    Read a YOLO labels file and return a list of dicts describing each IC detection
    that meets the confidence threshold, with its individual area in mm².
    """
    ic_records = []

    if not os.path.exists(labels_file) or os.path.getsize(labels_file) == 0:
        return ic_records

    ic_cls_ids = {
        cls_id
        for cls_id, cls_name in class_map.items()
        if cls_name.lower() in {"ic", "ics"}
    }

    ic_index = 0
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id not in ic_cls_ids:
                continue

            cls_name   = class_map[cls_id]
            confidence = float(parts[5]) if len(parts) > 5 else 1.0
            threshold  = conf_thresholds.get(cls_name, 0.5)

            if confidence < threshold:
                continue

            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm    = float(parts[3])
            height_norm   = float(parts[4])
            subcategory   = parts[6] if len(parts) > 6 else UNSPECIFIED

            width_mm_comp  = width_norm  * image_width_mm
            height_mm_comp = height_norm * image_height_mm
            area_mm2       = width_mm_comp * height_mm_comp

            ic_index += 1
            ic_records.append({
                'image':         image_label,
                'ic_index':      ic_index,
                'subcategory':   subcategory,
                'x_center_norm': round(x_center_norm, 6),
                'y_center_norm': round(y_center_norm, 6),
                'width_norm':    round(width_norm, 6),
                'height_norm':   round(height_norm, 6),
                'confidence':    round(confidence, 4),
                'width_mm':      round(width_mm_comp, 4),
                'height_mm':     round(height_mm_comp, 4),
                'area_mm2':      round(area_mm2, 4),
            })

    return ic_records


def main():
    parser = argparse.ArgumentParser(description="Generate Life Cycle Inventory from a PCB image")
    parser.add_argument('--input', required=True, nargs='+', help="Path(s) to input image(s).")
    parser.add_argument('--model', required=True, help="Path to the YOLO model (.pt) for detection")
    parser.add_argument('--output-xlsx', help="Output Excel file path (default: output/<filename>_metrics.xlsx)")
    parser.add_argument('--debug', action='store_true', help="Save debug images with detection rectangles")
    parser.add_argument('--human', action='store_true', help="Enable human-in-the-loop correction of detections (interactive GUI)")
    parser.add_argument('--metrics', action='store_true', help="Display detailed metrics for board and component dimensions after crop")
    parser.add_argument('--bbox-reduction', type=float, default=0.0, help="Shrink detection bounding boxes by percentage (e.g. 4 = shrink by 4%%)")
    parser.add_argument(
        '--bbox-uncertainty', type=float, default=0.0,
        help="Symmetric uncertainty on each bbox dimension in %% "
             "(e.g. 5 = ±5%%). Propagates into PI-based mass bounds."
    )
    parser.add_argument('--individual-ic', action='store_true',
                        help="Output the area of each individual IC detection to a separate "
                             "CSV/Excel file.")
    parser.add_argument('--taxonomy', default=None,
                        help="Subcategory definitions (default: config/taxonomy.yaml)")
    parser.add_argument('--impact-factors', default=None,
                        help="Impact factors per ecoinvent flow (default: config/impact_factors.yaml)")
    parser.add_argument('--gui-size', default="1360x820",
                        help="Size of the review window in pixels, WIDTHxHEIGHT (default: 1360x820)")
    args = parser.parse_args()

    try:
        gui_size = tuple(int(v) for v in args.gui_size.lower().split('x'))
        assert len(gui_size) == 2
    except (ValueError, AssertionError):
        print(f"Error: --gui-size must look like 1360x820 (got '{args.gui_size}').")
        sys.exit(1)

    taxonomy = Taxonomy.load(args.taxonomy)
    impact_factors = lci.ImpactFactors.load(args.impact_factors)

    image_paths = args.input
    if len(image_paths) > 1:
        print(f"\n📋 Multi-image mode: processing {len(image_paths)} images")
        for i, img_path in enumerate(image_paths, 1):
            print(f"   {i}. {img_path}")

    os.makedirs("output", exist_ok=True)
    os.makedirs("output/labels", exist_ok=True)

    title = pyfiglet.figlet_format('InventoryVision', font='ogre')
    print(f'[yellow]{title}[/yellow]')

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"Error: The file '{img_path}' does not exist.")
            sys.exit(1)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        if not img_path.lower().endswith(valid_extensions):
            print(f"Error: The file '{img_path}' does not appear to be a supported image format.")
            sys.exit(1)

    if not args.model:
        print(f"Error: No detection model detected.")
        sys.exit(1)

    print(f"Loading YOLO model: {args.model}")
    yolo_model = YOLO(args.model)
    model_class_names = yolo_model.names

    print(f"Model classes: {model_class_names}")

    default_class_map = {}
    for cls_id, cls_name in model_class_names.items():
        default_class_map[cls_id] = cls_name

    unmatched = [n for n in default_class_map.values() if taxonomy.macro(n) is None]
    if unmatched:
        print(f"⚠️  Model classes without a taxonomy entry (no subcategories, no LCI flow): {unmatched}")

    conf_thresholds = {}
    for cls_name in default_class_map.values():
        if cls_name.lower() in ["ic", "ics"]:
            conf_thresholds[cls_name] = 0.7
        elif cls_name.lower() in ["connector", "connectors"]:
            conf_thresholds[cls_name] = 0.6
        elif cls_name.lower() in ["resistor", "resistors", "capacitor", "capacitors"]:
            conf_thresholds[cls_name] = 0.5
        elif cls_name.lower() in ["inductor", "inductors"]:
            conf_thresholds[cls_name] = 0.4
        elif cls_name.lower() in ["diode", "diodes"]:
            conf_thresholds[cls_name] = 0.3
        elif cls_name.lower() in ["transistor", "transistors", "led", "leds"]:
            conf_thresholds[cls_name] = 0.2
        else:
            conf_thresholds[cls_name] = 0.5

    print("\n" + "="*60)
    print("CONFIDENCE THRESHOLD CONFIGURATION")
    print("="*60)
    for cls_name, threshold in conf_thresholds.items():
        print(f"  {cls_name:15s} : {threshold}")
    print("="*60 + "\n")

    all_total_mass         = []
    all_total_ic_area      = []
    all_counts_per_class   = []
    all_uncertainties      = []
    all_individual_ic_areas = []
    all_modifications_per_class = []
    all_lci_records = []          # one record per detection, all images (subcategory LCI)

    for idx, board_image_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"Processing image {idx+1}/{len(image_paths)}: {board_image_path}")
        print(f"{'='*60}\n")

        filename = os.path.splitext(os.path.basename(board_image_path))[0]
        extension = os.path.splitext(board_image_path)[1]

        if len(image_paths) > 1:
            filename_with_side = f"{filename}_img{idx+1}"
        else:
            filename_with_side = filename

        ##-- COMPONENT DETECTION AND CLASSIFICATION --
        detector = detection_classification.PCBDetector(
            args.model,
            bbox_reduction=args.bbox_reduction
        )

        state_file = f"output/labels/{filename}_state.json"

        skip_inference = os.path.exists(state_file) and args.human
        if skip_inference:
            print(f"📁 Found existing state file: {state_file}")
            print("   Skipping inference - will load state from file")
            detection_file = f"output/labels/{filename}.txt"
            if not os.path.exists(detection_file):
                with open(detection_file, 'w') as f:
                    pass
        else:
            print("Running component detection on full image...")
            detector.run(
                source=board_image_path,
                output_dir="output",
                conf_thresh=conf_thresholds,
                annotate=True
            )

        detection_file = f"output/labels/{filename}.txt"

        if not os.path.exists(detection_file):
            print(f"⚠️ Warning: No detection file found at {detection_file}")
            all_total_mass.append({})
            all_total_ic_area.append(0)
            all_counts_per_class.append({})
            all_uncertainties.append({
                'per_class': {},
                'total_nom': 0.0,
                'total_model_low': 0.0, 'total_model_high': 0.0,
                'total_bbox_low':  0.0, 'total_bbox_high':  0.0,
                'total_combined_low': 0.0, 'total_combined_high': 0.0,
            })
            all_individual_ic_areas.append([])
            all_modifications_per_class.append({})
            continue

        # In --human mode an empty file is not a dead end: boxes can still be added by hand.
        if os.path.getsize(detection_file) == 0 and not args.human:
            print(f"⚠️ Warning: Detection file is empty (no components detected)")
            all_total_mass.append({})
            all_total_ic_area.append(0)
            all_counts_per_class.append({})
            all_uncertainties.append({
                'per_class': {},
                'total_nom': 0.0,
                'total_model_low': 0.0, 'total_model_high': 0.0,
                'total_bbox_low':  0.0, 'total_bbox_high':  0.0,
                'total_combined_low': 0.0, 'total_combined_high': 0.0,
            })
            all_individual_ic_areas.append([])
            all_modifications_per_class.append({})
            continue

        class_counts_full, low_conf_full = analyze_detection_file(
            detection_file,
            default_class_map,
            conf_thresholds,
            title="DETECTION SUMMARY - FULL IMAGE (>= THRESHOLD)"
        )

        ##-- HUMAN-IN-THE-LOOP CORRECTION --
        if args.human:
            try:
                import numpy as np

                from scripts.human_correction import HumanCorrectionGUI
            except Exception as e:
                print(f"Error importing human correction GUI: {e}")
                print("Skipping human correction.")
            else:
                # The GUI reloads the state file itself when it exists (it keeps the
                # history of corrections); the label file is the starting point otherwise.
                detections = read_labels(detection_file)
                if os.path.exists(state_file):
                    print(f"📁 Resuming review from state file: {state_file}")

                print("\n" + "="*60)
                print("STARTING HUMAN INTERFACE (--human mode)")
                print("="*60 + "\n")

                gui = HumanCorrectionGUI(
                    image_path=board_image_path,
                    detections=detections,
                    class_map=default_class_map,
                    taxonomy=taxonomy,
                    state_file=state_file,
                    window_size=gui_size,
                )

                corrected = gui.run()

                try:
                    write_labels(detection_file, corrected)
                    n_sub = sum(1 for d in corrected if d.get("sub", UNSPECIFIED) != UNSPECIFIED)
                    print(f"Human corrections saved to detection file "
                          f"({len(corrected)} boxes, {n_sub} with a subcategory).")
                except Exception as e:
                    print(f"Error while saving corrected detections: {e}")



        ##-- CROP FOR DIMENSION MEASUREMENT --
        board_image_cropped_path = os.path.join("output", filename_with_side + "_cropped" + extension)
        crop_data_file = f"output/labels/{filename}_crop.json"

        crop_points = None
        height_mm   = None
        width_mm    = None
        loaded_crop = False

        if os.path.exists(crop_data_file):
            try:
                with open(crop_data_file, 'r') as f:
                    saved_crop = json.load(f)
                saved_points = saved_crop.get('crop_points')
                saved_height = saved_crop.get('height_mm')
                saved_width  = saved_crop.get('width_mm')

                print(f"\n📁 Found existing crop data: {crop_data_file}")
                print(f"   Crop points : {saved_points}")
                print(f"   Dimensions  : {saved_width} mm (W) x {saved_height} mm (H)")

                reuse = ''
                while reuse.lower() not in ['y', 'n']:
                    reuse = input("   Reuse saved crop and dimensions? (y/n): ").strip()

                if reuse.lower() == 'y':
                    crop_points = [tuple(p) for p in saved_points]
                    height_mm   = saved_height
                    width_mm    = saved_width
                    loaded_crop = True
                    print("   ✅ Using saved crop and dimensions.")

                    import numpy as np

                    from scripts.crop_board import crop_and_warp
                    image  = cv2.imread(board_image_path)
                    warped = crop_and_warp(image, crop_points)
                    cv2.imwrite(board_image_cropped_path, warped)
                    print(f"   ✅ Re-cropped image saved to: {board_image_cropped_path}")
                else:
                    print("   ↩ Will overwrite saved crop data with new input.")
            except Exception as e:
                print(f"⚠️ Could not load crop data ({e}), proceeding with new crop.")

        if not loaded_crop:
            mode = ''
            while mode.lower() not in ['terminal', 'gui']:
                mode = input("\nCropping - terminal or graphical interface? (terminal/gui): ")

            if mode.lower() == 'terminal':
                print("Enter 4 points manually as x y (top-left, top-right, bottom-right, bottom-left):")
                crop_points = []
                for i in ["TL", "TR", "BR", "BL"]:
                    coords = input(f"{i} (x y): ").split()
                    crop_points.append((float(coords[0]), float(coords[1])))

                import numpy as np

                from scripts.crop_board import crop_and_warp
                image  = cv2.imread(board_image_path)
                warped = crop_and_warp(image, crop_points)
                cv2.imwrite(board_image_cropped_path, warped)
                print(f"Saved cropped image to: {board_image_cropped_path}")
            else:
                print("Performing manual cropping via graphical interface...")
                while True:
                    crop_points = crop_board.main(input_path=board_image_path, output_path=board_image_cropped_path, term=False)
                    if crop_points is not None:
                        break
                    retry = ''
                    while retry.lower() not in ['y', 'n']:
                        retry = input("Cropping cancelled. Retry? (y/n): ").strip()
                    if retry.lower() == 'n':
                        sys.exit("Aborted: no crop selected.")

            print("\nPlease enter the real-world dimensions of the PCB (in millimeters):")
            height_mm = ask_positive_float("Height (mm): ")
            width_mm  = ask_positive_float("Width (mm): ")

            print(f"\nRecorded dimensions: {height_mm} mm x {width_mm} mm")

            try:
                crop_data = {
                    'crop_points': [list(p) for p in crop_points],
                    'height_mm':   height_mm,
                    'width_mm':    width_mm
                }
                with open(crop_data_file, 'w') as f:
                    json.dump(crop_data, f, indent=2)
                print(f"   💾 Crop data saved to: {crop_data_file}")
            except Exception as e:
                print(f"⚠️ Could not save crop data: {e}")

        ##-- TRANSFORM DETECTIONS TO CROPPED SPACE --
        original_img = cv2.imread(board_image_path)
        original_height, original_width = original_img.shape[:2]

        input_labels  = detection_file
        output_labels = f"output/labels/{filename_with_side}_cropped.txt"

        print("\nTransforming detections from full image to cropped space...")
        n_transformed = transform_yolo_detections(
            input_labels_file=input_labels,
            output_labels_file=output_labels,
            crop_points=crop_points,
            original_img_width=original_width,
            original_img_height=original_height,
            debug_image_path=board_image_cropped_path,
            class_names=default_class_map,
            original_image_path=board_image_path,
            min_overlap_threshold=0.05
        )

        print(f"✓ {n_transformed} detections transformed to cropped space")

        if n_transformed == 0:
            print(f"⚠️ Warning: No detections in cropped area")
            all_total_mass.append({})
            all_total_ic_area.append(0)
            all_counts_per_class.append({})
            all_uncertainties.append({
                'per_class': {},
                'total_nom': 0.0,
                'total_model_low': 0.0, 'total_model_high': 0.0,
                'total_bbox_low':  0.0, 'total_bbox_high':  0.0,
                'total_combined_low': 0.0, 'total_combined_high': 0.0,
            })
            all_individual_ic_areas.append([])
            all_modifications_per_class.append({})
            continue

        class_counts_cropped, low_conf_cropped = analyze_detection_file(
            output_labels,
            default_class_map,
            conf_thresholds,
            title="DETECTION SUMMARY - AFTER CROP (>= THRESHOLD)"
        )

        print("\n" + "="*60)
        print("BEFORE/AFTER CROP COMPARISON")
        print("="*60)

        all_classes = set(list(class_counts_full.keys()) + list(class_counts_cropped.keys()))
        for cls_name in sorted(all_classes):
            count_before = class_counts_full.get(cls_name, 0)
            count_after  = class_counts_cropped.get(cls_name, 0)
            delta = count_after - count_before
            if delta == 0:
                status = "="
            elif delta > 0:
                status = f"+{delta} (⚠️ added?)"
            else:
                status = f"{delta} (lost)"
            print(f"  {cls_name:15s}: {count_before:3d} → {count_after:3d} ({status})")

        total_before = sum(class_counts_full.values())
        total_after  = sum(class_counts_cropped.values())
        print(f"\n  {'TOTAL':15s}: {total_before:3d} → {total_after:3d} (delta: {total_after - total_before:+d})")
        print("="*60 + "\n")

        ##-- DISPLAY METRICS --
        if args.metrics:
            print("\n" + "="*60)
            print("DETAILED METRICS - COMPONENT DIMENSIONS")
            print("="*60)
            print(f"Board dimensions       : {width_mm:.2f} mm x {height_mm:.2f} mm")
            print(f"Board area             : {(width_mm * height_mm):.2f} mm²")
            print()

            cropped_img = cv2.imread(board_image_cropped_path)
            if cropped_img is not None:
                cropped_height_px, cropped_width_px = cropped_img.shape[:2]
                px_to_mm_width  = width_mm  / cropped_width_px
                px_to_mm_height = height_mm / cropped_height_px

                print(f"Cropped image          : {cropped_width_px} x {cropped_height_px} pixels")
                print(f"Conversion             : {px_to_mm_width:.4f} mm/px (width), {px_to_mm_height:.4f} mm/px (height)")
                print()

                metrics_img = cropped_img.copy()

                if os.path.exists(output_labels):
                    components_by_class = {}

                    with open(output_labels, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id        = int(parts[0])
                                cls_name      = default_class_map.get(cls_id, f"unknown_{cls_id}")
                                x_center_norm = float(parts[1])
                                y_center_norm = float(parts[2])
                                width_norm    = float(parts[3])
                                height_norm   = float(parts[4])

                                x_center_px = x_center_norm * cropped_width_px
                                y_center_px = y_center_norm * cropped_height_px
                                width_px    = width_norm    * cropped_width_px
                                height_px   = height_norm   * cropped_height_px

                                width_mm_comp  = width_px  * px_to_mm_width
                                height_mm_comp = height_px * px_to_mm_height
                                area_mm2       = width_mm_comp * height_mm_comp

                                x1 = int(x_center_px - width_px / 2)
                                y1 = int(y_center_px - height_px / 2)
                                x2 = int(x_center_px + width_px / 2)
                                y2 = int(y_center_px + height_px / 2)

                                cv2.rectangle(metrics_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                dim_text  = f"{width_mm_comp:.1f}x{height_mm_comp:.1f}mm"
                                area_text = f"{area_mm2:.1f}mm2"

                                font       = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.4
                                thickness  = 1

                                (text_width, text_height), baseline = cv2.getTextSize(dim_text, font, font_scale, thickness)
                                text_y = y1 - 5 if y1 > text_height + 10 else y2 + text_height + 5

                                cv2.rectangle(metrics_img,
                                            (x1, text_y - text_height - 3),
                                            (x1 + text_width + 5, text_y + 3),
                                            (0, 0, 0), -1)
                                cv2.putText(metrics_img, dim_text, (x1 + 2, text_y),
                                          font, font_scale, (255, 255, 255), thickness)

                                text_y2 = text_y + text_height + 5
                                (text_width2, text_height2), _ = cv2.getTextSize(area_text, font, font_scale, thickness)
                                cv2.rectangle(metrics_img,
                                            (x1, text_y2 - text_height2 - 3),
                                            (x1 + text_width2 + 5, text_y2 + 3),
                                            (0, 0, 0), -1)
                                cv2.putText(metrics_img, area_text, (x1 + 2, text_y2),
                                          font, font_scale, (255, 255, 255), thickness)

                                if cls_name not in components_by_class:
                                    components_by_class[cls_name] = []
                                components_by_class[cls_name].append({
                                    'width_mm':  width_mm_comp,
                                    'height_mm': height_mm_comp,
                                    'area_mm2':  area_mm2
                                })

                    metrics_image_path = os.path.join("output", filename_with_side + "_metrics" + extension)
                    cv2.imwrite(metrics_image_path, metrics_img)
                    print(f"✓ Annotated image saved: {metrics_image_path}")
                    print()
                    print("COMPONENT STATISTICS BY CLASS:")
                    print("-" * 60)

                    for cls_name in sorted(components_by_class.keys()):
                        components = components_by_class[cls_name]
                        count = len(components)
                        widths  = [c['width_mm']  for c in components]
                        heights = [c['height_mm'] for c in components]
                        areas   = [c['area_mm2']  for c in components]
                        avg_width  = sum(widths)  / count
                        avg_height = sum(heights) / count
                        avg_area   = sum(areas)   / count
                        total_area = sum(areas)
                        print(f"\n{cls_name} ({count} components) :")
                        print(f"  Average dimensions   : {avg_width:.2f} x {avg_height:.2f} mm")
                        print(f"  Average area         : {avg_area:.2f} mm²")
                        print(f"  Total area           : {total_area:.2f} mm²")
                        print(f"  Min/max area         : {min(areas):.2f} / {max(areas):.2f} mm²")

                    print("\n" + "="*60 + "\n")

        ##-- COMPONENT AREA TO MASS --
        total_mass, counts_per_class, total_ic_area, uncertainty, modifications_per_class = \
            component_to_mass.calculate_mass_per_class(
                component_file=output_labels,
                class_map=default_class_map,
                image_width_mm=width_mm,
                image_height_mm=height_mm,
                bbox_uncertainty_pct=args.bbox_uncertainty,
                state_file=state_file,
            )

        all_total_mass.append(total_mass)
        all_total_ic_area.append(total_ic_area)
        all_counts_per_class.append(counts_per_class)
        all_uncertainties.append(uncertainty)
        all_modifications_per_class.append(modifications_per_class)

        ##-- SUBCATEGORY LCI RECORDS (die area per package type, flows, impacts) --
        all_lci_records.extend(lci.records_from_labels(
            label_file=output_labels,
            class_map=default_class_map,
            taxonomy=taxonomy,
            width_mm=width_mm,
            height_mm=height_mm,
            bbox_uncertainty_pct=args.bbox_uncertainty,
            image_label=filename_with_side,
        ))

        ##-- INDIVIDUAL IC AREAS --
        if args.individual_ic:
            image_label = filename_with_side
            ic_records = extract_individual_ic_areas(
                labels_file=output_labels,
                class_map=default_class_map,
                conf_thresholds=conf_thresholds,
                image_width_mm=width_mm,
                image_height_mm=height_mm,
                image_label=image_label,
            )
            all_individual_ic_areas.append(ic_records)

            print("\n" + "="*60)
            print(f"INDIVIDUAL IC AREAS — {image_label}")
            print("="*60)
            if ic_records:
                for rec in ic_records:
                    print(
                        f"  IC #{rec['ic_index']:3d} | "
                        f"{rec['width_mm']:.3f} x {rec['height_mm']:.3f} mm | "
                        f"area = {rec['area_mm2']:.4f} mm² | "
                        f"conf = {rec['confidence']:.3f}"
                    )
                total = sum(r['area_mm2'] for r in ic_records)
                print(f"  {'─'*50}")
                print(f"  Total ({len(ic_records)} ICs) : {total:.4f} mm²")

                cropped_img_annot = cv2.imread(board_image_cropped_path)
                if cropped_img_annot is not None:
                    annot_h, annot_w = cropped_img_annot.shape[:2]
                    font          = cv2.FONT_HERSHEY_DUPLEX
                    box_color     = (0, 140, 255)
                    text_bg_color = (0, 140, 255)
                    text_color    = (255, 255, 255)

                    for rec in ic_records:
                        xc = rec['x_center_norm'] * annot_w
                        yc = rec['y_center_norm'] * annot_h
                        bw = rec['width_norm']    * annot_w
                        bh = rec['height_norm']   * annot_h
                        x1 = max(0, int(xc - bw / 2))
                        y1 = max(0, int(yc - bh / 2))
                        x2 = min(annot_w - 1, int(xc + bw / 2))
                        y2 = min(annot_h - 1, int(yc + bh / 2))

                        cv2.rectangle(cropped_img_annot, (x1, y1), (x2, y2), box_color, 2)

                        font_scale = max(0.5, min(2.5, bw / 120))
                        label      = str(rec['ic_index'])
                        thickness  = max(1, int(font_scale * 1.8))
                        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        pad = 4
                        lx  = x1 + pad
                        ly  = y1 + th + pad
                        if th + 2 * pad > (y2 - y1):
                            ly = max(th + pad, y1 - pad)

                        cv2.rectangle(cropped_img_annot,
                                      (lx - pad, ly - th - pad),
                                      (lx + tw + pad, ly + baseline + pad),
                                      text_bg_color, -1)
                        cv2.putText(cropped_img_annot, label,
                                    (lx, ly), font, font_scale,
                                    text_color, thickness, cv2.LINE_AA)

                    ic_annot_path = os.path.join("output", filename_with_side + "_ic_annotated" + extension)
                    cv2.imwrite(ic_annot_path, cropped_img_annot)
                    print(f"\n✅ Annotated IC image saved to: {ic_annot_path}")
            else:
                print("  No ICs detected above threshold.")
            print("="*60 + "\n")
        else:
            all_individual_ic_areas.append([])

    ##-- GENERATE EXCEL OUTPUT --
    component_to_flow = {
        "Capacitor": "market for capacitor, for surface-mounting",
        "capacitor": "market for capacitor, for surface-mounting",
        "capacitors": "market for capacitor, for surface-mounting",
        "Diode": "market for diode, glass-, for surface-mounting",
        "diode": "market for diode, glass-, for surface-mounting",
        "diodes": "market for diode, glass-, for surface-mounting",
        "LED": "market for light emitting diode",
        "led": "market for light emitting diode",
        "leds": "market for light emitting diode",
        "Inductor": "market for inductor, low value multilayer chip",
        "inductor": "market for inductor, low value multilayer chip",
        "inductors": "market for inductor, low value multilayer chip",
        "Transistor": "market for transistor, surface-mounted",
        "transistor": "market for transistor, surface-mounted",
        "transistors": "market for transistor, surface-mounted",
        "Resistor": "market for resistor, surface-mounted",
        "resistor": "market for resistor, surface-mounted",
        "resistors": "market for resistor, surface-mounted",
        "Connector": "market for electric connector, peripheral type buss",
        "connector": "market for electric connector, peripheral type buss",
        "connectors": "market for electric connector, peripheral type buss",
        "IC": "market for integrated circuit, logic type",
        "ic": "market for integrated circuit, logic type",
        "ICs": "market for integrated circuit, logic type",
        "ics": "market for integrated circuit, logic type"
    }

    # Merge results from all images
    merged_mass   = {}
    merged_ic_area = sum(all_total_ic_area)
    merged_counts = {}

    for total_mass in all_total_mass:
        for component_class, mass_mg in total_mass.items():
            merged_mass[component_class] = merged_mass.get(component_class, 0) + mass_mg

    for counts_per_class in all_counts_per_class:
        for component_class, count in counts_per_class.items():
            merged_counts[component_class] = merged_counts.get(component_class, 0) + count

    # Merge uncertainties across images (quadratic combination per source)
    merged_per_class = {}   # cls -> {nom, model_low, model_high, bbox_low, bbox_high, combined_low, combined_high}
    merged_total_nom = 0.0

    # Quadratic accumulators for board-level totals
    sq_model_lo = sq_model_hi = 0.0
    sq_bbox_lo  = sq_bbox_hi  = 0.0
    sq_comb_lo  = sq_comb_hi  = 0.0

    for unc in all_uncertainties:
        merged_total_nom += unc.get('total_nom', 0.0)
        for cls, u in unc.get('per_class', {}).items():
            if cls not in merged_per_class:
                merged_per_class[cls] = {
                    'nom': 0.0,
                    'model_low': 0.0, 'model_high': 0.0,
                    'bbox_low':  0.0, 'bbox_high':  0.0,
                    'combined_low': 0.0, 'combined_high': 0.0,
                }
            merged_per_class[cls]['nom']           += u['nom']
            merged_per_class[cls]['model_low']     += u['model_low']
            merged_per_class[cls]['model_high']    += u['model_high']
            merged_per_class[cls]['bbox_low']      += u['bbox_low']
            merged_per_class[cls]['bbox_high']     += u['bbox_high']
            merged_per_class[cls]['combined_low']  += u['combined_low']
            merged_per_class[cls]['combined_high'] += u['combined_high']

    import math
    for cls, u in merged_per_class.items():
        sq_model_lo += (u['nom'] - u['model_low'])  ** 2
        sq_model_hi += (u['model_high'] - u['nom']) ** 2
        sq_bbox_lo  += (u['nom'] - u['bbox_low'])   ** 2
        sq_bbox_hi  += (u['bbox_high'] - u['nom'])  ** 2
        sq_comb_lo  += (u['nom'] - u['combined_low'])  ** 2
        sq_comb_hi  += (u['combined_high'] - u['nom']) ** 2

    merged_total_model_low  = max(0.0, merged_total_nom - math.sqrt(sq_model_lo))
    merged_total_model_high = merged_total_nom + math.sqrt(sq_model_hi)
    merged_total_bbox_low   = max(0.0, merged_total_nom - math.sqrt(sq_bbox_lo))
    merged_total_bbox_high  = merged_total_nom + math.sqrt(sq_bbox_hi)
    merged_total_comb_low   = max(0.0, merged_total_nom - math.sqrt(sq_comb_lo))
    merged_total_comb_high  = merged_total_nom + math.sqrt(sq_comb_hi)

    # Build Excel rows
    # Columns:
    #   name | count | value (nom)
    #   | mass_model_low | mass_model_high   <- model PI only
    #   | mass_bbox_low  | mass_bbox_high    <- bbox uncertainty only
    #   | mass_low       | mass_high         <- quadratic combination (both sources)
    excel_data = []

    for component_class, mass_mg in merged_mass.items():
        flow_name = component_to_flow.get(component_class, f"market for {component_class}")
        count = merged_counts.get(component_class, 0)
        u = merged_per_class.get(component_class, {})
        nom        = u.get('nom',           mass_mg)
        model_low  = u.get('model_low',     nom)
        model_high = u.get('model_high',    nom)
        bbox_low   = u.get('bbox_low',      nom)
        bbox_high  = u.get('bbox_high',     nom)
        comb_low   = u.get('combined_low',  nom)
        comb_high  = u.get('combined_high', nom)

        excel_data.append({
            'name':             flow_name,
            'count':            count,
            'value':            round(nom,        6),
            'mass_model_low':   round(model_low,  6),
            'mass_model_high':  round(model_high, 6),
            'mass_bbox_low':    round(bbox_low,   6),
            'mass_bbox_high':   round(bbox_high,  6),
            'mass_low':         round(comb_low,   6),
            'mass_high':        round(comb_high,  6),
        })

    if merged_ic_area > 0:
        ic_flow  = component_to_flow.get("IC", "market for integrated circuit, logic type")
        ic_count = merged_counts.get("IC", merged_counts.get("ic",
                   merged_counts.get("ICs", merged_counts.get("ics", 0))))
        excel_data.append({
            'name':            ic_flow,
            'count':           ic_count,
            'value':           merged_ic_area,
            'mass_model_low':  merged_ic_area,
            'mass_model_high': merged_ic_area,
            'mass_bbox_low':   merged_ic_area,
            'mass_bbox_high':  merged_ic_area,
            'mass_low':        merged_ic_area,
            'mass_high':       merged_ic_area,
        })

    # Summary row
    excel_data.append({
        'name':            'TOTAL',
        'count':           sum(row['count'] for row in excel_data),
        'value':           round(merged_total_nom,        4),
        'mass_model_low':  round(merged_total_model_low,  4),
        'mass_model_high': round(merged_total_model_high, 4),
        'mass_bbox_low':   round(merged_total_bbox_low,   4),
        'mass_bbox_high':  round(merged_total_bbox_high,  4),
        'mass_low':        round(merged_total_comb_low,   4),
        'mass_high':       round(merged_total_comb_high,  4),
    })

    df = pd.DataFrame(excel_data)

    # Determine output path
    if args.output_xlsx:
        output_xlsx = args.output_xlsx
    else:
        base_filename = os.path.splitext(os.path.basename(image_paths[0]))[0]
        base_filename = (base_filename
                         .replace('_front', '').replace('_back', '')
                         .replace('_top', '').replace('_bottom', '')
                         .replace('_side1', '').replace('_side2', ''))
        output_xlsx = os.path.join("output", f"{base_filename}_metrics.xlsx")

    with pd.ExcelWriter(output_xlsx) as writer:
        # Original sheet, unchanged (one row per macro class)
        df.to_excel(writer, index=False, sheet_name='Component Metrics')
        # New sheets: LCI by subcategory, per-component records, impacts, assumptions
        lci_rows, lci_totals, lci_missing = lci.write_excel_sheets(
            writer, all_lci_records, taxonomy, impact_factors, args.bbox_uncertainty)

    lci.print_summary(lci_rows, lci_totals, lci_missing, impact_factors)

    print(f"\n{'='*60}")
    print(f"✅ Component metrics saved to: {output_xlsx}")
    print("   - Sheets: Component Metrics, LCI by subcategory, Components, Impacts, Assumptions")
    if len(image_paths) > 1:
        print(f"   - Merged results from {len(image_paths)} images")
    print(f"   - ICs: total area = {merged_ic_area:.3f} mm²")
    print("   - Connectors: default mass calculation")
    print(f"{'='*60}")
    print(f"\n  Total mass (nominal) : {merged_total_nom:.2f} mg")
    print(f"  Model PI only        : [{merged_total_model_low:.2f} – {merged_total_model_high:.2f}] mg")
    print(f"  BBox uncertainty only: [{merged_total_bbox_low:.2f} – {merged_total_bbox_high:.2f}] mg")
    print(f"  Combined (quadratic) : [{merged_total_comb_low:.2f} – {merged_total_comb_high:.2f}] mg")
    if args.bbox_uncertainty > 0:
        print(f"  (includes ±{args.bbox_uncertainty:.1f}% bbox size uncertainty)")

    ##-- INDIVIDUAL IC AREAS OUTPUT FILE --
    if args.individual_ic:
        all_records = [rec for records in all_individual_ic_areas for rec in records]

        if all_records:
            ic_df = pd.DataFrame(all_records, columns=[
                'image', 'ic_index', 'subcategory',
                'x_center_norm', 'y_center_norm',
                'width_norm', 'height_norm',
                'confidence',
                'width_mm', 'height_mm', 'area_mm2'
            ])
            ic_df.insert(0, 'global_index', range(1, len(ic_df) + 1))

            total_area_all = ic_df['area_mm2'].sum()
            summary_row = pd.DataFrame([{
                'global_index': '',
                'image': 'TOTAL',
                'ic_index': len(ic_df),
                'subcategory': '',
                'x_center_norm': '', 'y_center_norm': '',
                'width_norm': '', 'height_norm': '',
                'confidence': '',
                'width_mm': '', 'height_mm': '',
                'area_mm2': round(total_area_all, 4),
            }])
            ic_df_with_total = pd.concat([ic_df, summary_row], ignore_index=True)

            base_xlsx_noext = os.path.splitext(output_xlsx)[0]
            if base_xlsx_noext.endswith('_metrics'):
                base_xlsx_noext = base_xlsx_noext[:-len('_metrics')]
            ic_output_xlsx = f"{base_xlsx_noext}_ic_areas.xlsx"

            ic_df_with_total.to_excel(ic_output_xlsx, index=False, sheet_name='Individual IC Areas')

            print(f"\n{'='*60}")
            print(f"✅ Individual IC areas saved to: {ic_output_xlsx}")
            print(f"   - {len(ic_df)} IC(s) across {len(image_paths)} image(s)")
            print(f"   - Total IC area: {total_area_all:.4f} mm²")
            print(f"{'='*60}")
        else:
            print(f"\n⚠️ --individual-ic: no IC detections found above threshold across all images.")


if __name__ == "__main__":
    main()