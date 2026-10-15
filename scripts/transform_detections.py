import cv2
import numpy as np

def get_perspective_transform_from_points(src_points):
    """
    Calculate perspective transform matrix from 4 corner points.
    
    Args:
        src_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    Order: TL, TR, BR, BL
    
    Returns:
        Transform matrix, destination width, destination height
    """
    pts = np.array(src_points, dtype="float32")
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],      # top-left
        pts[np.argmin(diff)],   # top-right
        pts[np.argmax(s)],      # bottom-right
        pts[np.argmax(diff)]    # bottom-left
    ], dtype="float32")
    
    (tl, tr, br, bl) = rect
    
    # Calculate width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return M, maxWidth, maxHeight


def transform_yolo_detections(input_labels_file, output_labels_file, 
                              crop_points, original_img_width, original_img_height,
                              debug_image_path=None, class_names=None, original_image_path=None,
                              min_overlap_threshold=0.05):
    """
    Transform YOLO detection coordinates from original image to cropped space.
    
    MODIFIED: Keep full bounding boxes if they overlap > 5% with crop area,
    instead of clipping them to crop boundaries.
    
    Args:
        input_labels_file: Path to original YOLO labels file
        output_labels_file: Path to save transformed labels
        crop_points: List of 4 corner points in original image [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        original_img_width: Width of original image in pixels
        original_img_height: Height of original image in pixels
        debug_image_path: Optional path to save debug visualization with rectangles
        class_names: Optional dict mapping class IDs to names (e.g., {0: 'resistors', 1: 'capacitors'})
        original_image_path: Optional path to original image for full debug visualization
        min_overlap_threshold: Minimum overlap ratio to keep detection (default: 0.05 = 5%)
    
    Returns:
        int: Number of detections successfully transformed
    """
    # Get perspective transform
    M, crop_width, crop_height = get_perspective_transform_from_points(crop_points)
    
    transformed_lines = []
    skipped_outside = 0
    skipped_low_overlap = 0
    kept_partial = 0
    total_input = 0
    
    print(f"\n📐 Transformation geometry:")
    print(f"   Original image: {original_img_width}x{original_img_height} px")
    print(f"   Cropped image: {crop_width}x{crop_height} px")
    print(f"   Minimum overlap: {min_overlap_threshold*100:.0f}%")
    
    with open(input_labels_file, 'r') as f:
        for line in f:
            total_input += 1
            parts = line.strip().split()
            
            # Support both formats: with and without confidence
            if len(parts) < 5:
                print(f"⚠️  Skipping invalid line (not enough values): {line.strip()}")
                continue
            
            class_id = parts[0]
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            confidence = float(parts[5]) if len(parts) > 5 else None
            # Extra columns (e.g. the subcategory written by the review GUI) are kept as-is
            extra = (" " + " ".join(parts[6:])) if len(parts) > 6 else ""

            # Convert normalized coordinates to absolute pixels in original image
            x_center = x_center_norm * original_img_width
            y_center = y_center_norm * original_img_height
            box_width = width_norm * original_img_width
            box_height = height_norm * original_img_height
            
            # Get 4 corners of bounding box in original image
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2
            
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype='float32').reshape(-1, 1, 2)
            
            # Transform corners to cropped space
            transformed_corners = cv2.perspectiveTransform(corners, M)
            transformed_corners = transformed_corners.reshape(-1, 2)
            
            # Calculate new bounding box from transformed corners
            x_coords = transformed_corners[:, 0]
            y_coords = transformed_corners[:, 1]
            
            new_x1 = np.min(x_coords)
            new_y1 = np.min(y_coords)
            new_x2 = np.max(x_coords)
            new_y2 = np.max(y_coords)
            
            # Check if box is completely outside cropped image bounds
            if new_x2 < 0 or new_y2 < 0 or new_x1 > crop_width or new_y1 > crop_height:
                skipped_outside += 1
                continue
            
            # Calculate overlap with crop area (before clipping)
            clipped_x1 = max(0, new_x1)
            clipped_y1 = max(0, new_y1)
            clipped_x2 = min(crop_width, new_x2)
            clipped_y2 = min(crop_height, new_y2)
            
            overlap_area = (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)
            original_box_area = (new_x2 - new_x1) * (new_y2 - new_y1)
            
            if original_box_area <= 0:
                skipped_low_overlap += 1
                continue
            
            overlap_ratio = overlap_area / original_box_area
            
            # If overlap is less than threshold, skip this detection
            if overlap_ratio < min_overlap_threshold:
                skipped_low_overlap += 1
                continue
            
            # MODIFIED: Keep full box if overlap >= threshold
            # Do NOT clip to crop boundaries
            final_x1 = new_x1
            final_y1 = new_y1
            final_x2 = new_x2
            final_y2 = new_y2
            
            # Track if box extends beyond crop
            extends_beyond = (final_x1 < 0 or final_y1 < 0 or 
                            final_x2 > crop_width or final_y2 > crop_height)
            if extends_beyond:
                kept_partial += 1
            
            # Convert back to YOLO normalized format
            # Use crop dimensions for normalization
            new_x_center = (final_x1 + final_x2) / 2 / crop_width
            new_y_center = (final_y1 + final_y2) / 2 / crop_height
            new_width = (final_x2 - final_x1) / crop_width
            new_height = (final_y2 - final_y1) / crop_height
            
            # Write with confidence if it was present
            if confidence is not None:
                transformed_lines.append(
                    f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f} {confidence:.6f}{extra}\n"
                )
            else:
                transformed_lines.append(
                    f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}{' 1.000000' + extra if extra else ''}\n"
                )
    
    # Write transformed labels
    with open(output_labels_file, 'w') as f:
        f.writelines(transformed_lines)
    
    # Print detailed statistics
    print(f"\n📊 Transformation statistics:")
    print(f"   Input detections:        {total_input}")
    print(f"   Transformed:             {len(transformed_lines)}")
    print(f"   - Fully inside crop:     {len(transformed_lines) - kept_partial}")
    print(f"   - Partially outside:     {kept_partial} (kept full box)")
    print(f"   Skipped (outside):       {skipped_outside}")
    print(f"   Skipped (< {min_overlap_threshold*100:.0f}% overlap): {skipped_low_overlap}")
    print(f"   Success rate:            {len(transformed_lines)/max(total_input, 1)*100:.1f}%")
    
    # Create debug visualization of cropped image if requested
    if debug_image_path and transformed_lines:
        # Load the cropped image
        debug_img = cv2.imread(debug_image_path)
        if debug_img is not None:
            # Count detections per class
            class_counts = {}
            
            # Draw all transformed detections
            for line in transformed_lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * crop_width
                y_center = float(parts[2]) * crop_height
                width = float(parts[3]) * crop_width
                height = float(parts[4]) * crop_height
                confidence = float(parts[5]) if len(parts) > 5 else None
                
                # Count by class
                class_name = class_names.get(class_id, f"Class_{class_id}") if class_names else f"Class_{class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Calculate corners
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Check if extends beyond image
                extends = (x1 < 0 or y1 < 0 or x2 > crop_width or y2 > crop_height)
                
                # Draw rectangle with different colors per class
                color_map = {
                    0: (255, 0, 0),    # Blue
                    1: (0, 255, 0),    # Green
                    2: (0, 0, 255),    # Red
                    3: (255, 255, 0),  # Cyan
                    4: (255, 0, 255),  # Magenta
                    5: (0, 255, 255),  # Yellow
                    6: (128, 0, 128),  # Purple
                    7: (255, 128, 0),  # Orange
                    8: (0, 128, 255),  # Sky blue
                }
                color = color_map.get(class_id, (0, 255, 0))
                
                # Use thicker line for boxes that extend beyond
                thickness = 3 if extends else 2
                
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
                
                # Add class label with confidence
                if class_names and class_id in class_names:
                    label = class_names[class_id]
                else:
                    label = f"C{class_id}"
                
                if confidence is not None:
                    label = f"{label} {confidence:.2f}"
                    
                # Add text with black background for readability
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(debug_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                cv2.putText(debug_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add summary text at the top
            summary = f"Total: {len(transformed_lines)} detections ({kept_partial} partial)"
            cv2.putText(debug_img, summary, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save debug image
            debug_output_path = debug_image_path.replace('.jpg', '_debug.jpg').replace('.png', '_debug.png')
            cv2.imwrite(debug_output_path, debug_img)
            print(f"\n✓ Debug image saved: {debug_output_path}")
            
            # Print class breakdown
            print(f"\n📋 Detections by class (after crop):")
            for cls_name, count in sorted(class_counts.items()):
                print(f"   {cls_name:15s}: {count:3d}")
    
    return len(transformed_lines)