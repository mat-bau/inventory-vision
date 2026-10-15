import math

import cv2
import numpy as np

clicked_points = []
current_mouse_pos = None
dragged_idx = None
draw_scale = 1.0  # display scale of the window (display px / image px)

MAX_WIN_W, MAX_WIN_H = 1200, 800
KEY_ENTER = (10, 13)
KEY_BACKSPACE = (8, 127)
KEY_ESC = 27

def draw_shape(img, points, moving_point=None):
    """
    Draw the evolving shape (point, line, triangle, quadrilateral)
    using the clicked points and optionally the current mouse position.
    - Outline and points in red.
    - Transparent red fill inside the shape.

    Args:
        img (np.ndarray): the image to draw
        points (np.ndarray): the clicked points to draw
        moving_point (np.ndarray): the clicked points to draw
    """
    pts = points.copy()
    if moving_point is not None and len(pts) < 4 and dragged_idx is None:
        pts.append(moving_point)

    # Radius/thickness in image pixels so they stay visible once the window is shrunk
    r = max(5, int(5 / draw_scale))
    t = max(2, int(2 / draw_scale))

    overlay = img.copy()

    if len(pts) == 1:
        cv2.circle(img, pts[0], r, (0, 0, 255), -1)  # Red dot
    elif len(pts) >= 2:
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        if len(pts) > 2:
            cv2.fillPoly(overlay, [pts_np], color=(0, 0, 255))
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts_np], isClosed=len(pts) == 4, color=(0, 0, 255), thickness=t)
        for p in pts:
            cv2.circle(img, p, r, (0, 0, 255), -1)


def mouse_callback(event, x, y, flags, param):
    """
    Single mouse callback function for both clicks and mouse move.

    Args:
        event (int): the mouse event
        x (int): the x coordinate of clicked point
        y (int): the y coordinate of clicked point
        flags (int): the mouse event flags
        param (np.ndarray): the clicked point
    """
    global clicked_points, current_mouse_pos, dragged_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        # Grab an existing point if the click is close to it, otherwise add a new one
        grab_radius = 15 / draw_scale
        if clicked_points:
            idx = min(range(len(clicked_points)), key=lambda i: math.dist(clicked_points[i], (x, y)))
            if math.dist(clicked_points[idx], (x, y)) <= grab_radius:
                dragged_idx = idx
                return
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        current_mouse_pos = (x, y)
        if dragged_idx is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            clicked_points[dragged_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragged_idx = None

def order_points(pts):
    """
    Orders points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): the clicked points to draw
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def crop_and_warp(image, pts):
    """
    Warp the quadrilateral area defined by pts into a rectangle.

    Args:
        image (np.ndarray): the image to crop
        pts (np.ndarray): the clicked points to draw
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def _close_windows():
    """Destroy OpenCV windows and pump the event loop so they really close (macOS)."""
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


def main(input_path, output_path, term: bool = False):
    """
    Let the user select the 4 board corners, then save the warped crop.

    Returns:
        list[tuple] | None: the 4 points in original image pixels, or None if cancelled.
    """
    global clicked_points, current_mouse_pos, dragged_idx, draw_scale

    # RESET GLOBAL VARIABLES FOR EACH NEW IMAGE
    clicked_points = []
    current_mouse_pos = None
    dragged_idx = None

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    if term:
        # Terminal input mode
        print("Enter 4 points manually as x y (top-left, top-right, bottom-right, bottom-left):")
        pts = []
        for i in ["TL", "TR", "BR", "BL"]:
            coords = input(f"{i} (x y): ").split()
            pts.append((float(coords[0]), float(coords[1])))
    else:
        # GUI mode with dynamic shape preview
        clone = image.copy()
        H, W = image.shape[:2]
        draw_scale = min(1.0, MAX_WIN_W / W, MAX_WIN_H / H)

        window_name = "Crop board"
        preview_name = "Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(W * draw_scale), int(H * draw_scale))
        cv2.setWindowTitle(window_name, "Click 4 corners | Enter: validate  Backspace: undo  R: reset  Esc: cancel")
        # In WINDOW_NORMAL, OpenCV maps mouse coords back to original image pixels
        cv2.setMouseCallback(window_name, mouse_callback)

        print("Click the 4 board corners (drag a point to move it).")
        print("Enter: validate | Backspace: undo last point | R: reset | Esc: cancel")

        preview_shown = False
        cancelled = False
        while True:
            temp_img = clone.copy()
            draw_shape(temp_img, clicked_points, moving_point=current_mouse_pos)
            cv2.imshow(window_name, temp_img)

            if len(clicked_points) == 4 and dragged_idx is None:
                warped = crop_and_warp(image, clicked_points)
                if warped.size:
                    cv2.imshow(preview_name, warped)
                    if not preview_shown:
                        ph, pw = warped.shape[:2]
                        ps = min(1.0, 600 / pw, 600 / ph)
                        cv2.resizeWindow(preview_name, max(1, int(pw * ps)), max(1, int(ph * ps)))
                        preview_shown = True
            elif preview_shown and len(clicked_points) < 4:
                cv2.destroyWindow(preview_name)
                cv2.waitKey(1)
                preview_shown = False

            key = cv2.waitKey(20) & 0xFF
            if key in KEY_BACKSPACE:
                if clicked_points:
                    clicked_points.pop()
                dragged_idx = None
            elif key in (ord('r'), ord('R')):
                clicked_points.clear()
                dragged_idx = None
            elif key in KEY_ENTER:
                if len(clicked_points) == 4:
                    break
                print(f"Need 4 points to validate ({len(clicked_points)}/4).")
            elif key == KEY_ESC:
                cancelled = True
                break

        _close_windows()
        if cancelled:
            print("Cropping cancelled.")
            return None
        pts = list(clicked_points)

    warped = crop_and_warp(image, pts)
    cv2.imwrite(output_path, warped)
    print(f"Saved cropped image to: {output_path}")
    return pts
