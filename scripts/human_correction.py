"""
Human-in-the-loop review of the detections (``main.py --human``).

One window: image viewport on the left, control panel on the right, status
bar at the bottom. Everything works with the left button and the keyboard
(trackpad friendly); right-drag and middle-drag are optional shortcuts.

Modes
  MODIFY (M)  click a box to select it, then change its class (panel or 1-9)
              and its subcategory (panel or [ ]); drag it to move it, drag a
              handle to resize it; drag the background to pan.
  ADD (A)     drag to draw a new box; the new box is selected so its class and
              subcategory can be chosen right away. Click (no drag) selects.
  DELETE (D)  click a box to delete it.

Navigation
  wheel / + -        zoom around the cursor / the centre       F  fit image
  drag background    pan (MODIFY/DELETE); Shift-drag or right-drag in any mode
  arrow keys         pan
  N / P              next / previous detection, lowest confidence first

Editing
  X, Delete, Backspace  delete the selection      0  subcategory -> unspecified
  Z or Ctrl-Z  undo     Y or Ctrl-Y  redo          S  save
  B  show/hide boxes    T  show/hide labels        H or ?  help
  Esc  cancel / deselect / quit                    Q  quit (always saves)

State (``output/labels/<image>_state.json``) is saved after every change and
reloaded on the next run. Version-1 state files are still accepted.

Platform notes (OpenCV highgui backends behave differently):
  * macOS / Cocoa: scroll deltas arrive in (x, y), not in ``flags``; the
    original GUI read ``flags`` and therefore always zoomed out.
  * Linux / Qt: Qt zooms the window itself on every wheel event, on top of the
    application zoom, and cannot be told not to. The window is recreated after
    a wheel event to cancel it.
  * Cocoa only reports mouse events inside the window: a button released
    outside is detected on the next move and closes the drag.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np

from scripts.taxonomy import UNSPECIFIED, Taxonomy

def _highgui_backend():
    """'QT', 'COCOA', 'GTK3', 'WIN32', ... as reported by the OpenCV build."""
    m = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.M)
    return m.group(1).upper() if m else "UNKNOWN"


BACKEND = _highgui_backend()
if BACKEND.startswith("QT"):
    BACKEND = "QT"

WINDOW_NAME = "InventoryVision - review"
PANEL_W = 340
STATUS_H = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX
HANDLE_R = 6           # handle half-size (display px)
CLICK_TOL = 4          # px of movement below which a press is a click
MIN_BOX_PX = 4         # minimum size of a drawn box (display px)
ZOOM_STEP = 1.15
MAX_ZOOM = 8.0
HISTORY_MAX = 200

# High-visibility colours (BGR) - readable on green/blue/black boards.
PALETTE = [
    (255, 200, 0), (0, 140, 255), (60, 60, 255), (120, 255, 60), (0, 230, 255),
    (255, 90, 200), (200, 80, 255), (230, 255, 0), (60, 255, 180), (160, 160, 255),
]
SELECT_COLOR = (255, 255, 255)
DELETE_COLOR = (40, 40, 255)

# Panel colours
P_BG, P_TEXT, P_MUTED = (246, 246, 246), (30, 30, 30), (120, 120, 120)
P_ACCENT, P_BTN, P_LINE = (215, 120, 0), (226, 226, 226), (210, 210, 210)
VIEW_BG = (38, 38, 38)

# Special keys as returned by cv2.waitKeyEx (GTK/Qt, Windows, macOS).
K_LEFT = {65361, 2424832, 63234}
K_UP = {65362, 2490368, 63232}
K_RIGHT = {65363, 2555904, 63235}
K_DOWN = {65364, 2621440, 63233}
K_DELETE = {8, 127, 65535, 65288, 3014656, 63272}
K_ESC, K_CTRL_Z, K_CTRL_Y = 27, 26, 25

HELP_LINES = [
    "MODES   M modify    A add    D delete",
    "",
    "MODIFY  click a box to select it (click again to cycle overlapping boxes)",
    "        drag the box to move it, drag a white handle to resize it",
    "        class: panel or keys 1-9     subcategory: panel or [ ]  (0 = unspecified)",
    "ADD     drag to draw a box, then pick its class / subcategory in the panel",
    "DELETE  click a box to delete it    (any mode: X / Delete / Backspace)",
    "",
    "VIEW    wheel or + - zoom    F fit    arrows pan",
    "        drag background to pan (Shift-drag or right-drag in any mode)",
    "REVIEW  N / P next / previous detection, lowest confidence first",
    "",
    "Z undo   Y redo   S save   B boxes   T labels   Esc deselect   Q quit",
    "",
    "Every change is saved automatically.         press H or ? to close",
]


# =============================================================================
# Helpers
# =============================================================================
def _text(img, s, org, scale=0.45, color=P_TEXT, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def _text_w(s, scale=0.45, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def _fit_text(s, max_w, scale=0.45, thick=1):
    if _text_w(s, scale, thick) <= max_w:
        return s
    while s and _text_w(s + "..", scale, thick) > max_w:
        s = s[:-1]
    return s + ".."


def _ascii(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def _box_changed(a, b, eps=1e-6):
    return any(abs(a[k] - b[k]) > eps for k in ("xc", "yc", "w", "h"))


def load_state_detections(state_file):
    """Detections stored in a state file (v1 or v2) as label dicts, or [] on error."""
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for d in state.get("detections", []):
        if isinstance(d, dict):
            out.append({k: d.get(k) for k in ("cls", "xc", "yc", "w", "h", "conf")} |
                       {"sub": d.get("sub") or UNSPECIFIED})
        elif len(d) >= 5:
            out.append({"cls": int(d[0]), "xc": float(d[1]), "yc": float(d[2]), "w": float(d[3]),
                        "h": float(d[4]), "conf": float(d[5]) if len(d) > 5 else 1.0, "sub": UNSPECIFIED})
    return out


# =============================================================================
# GUI
# =============================================================================
class HumanCorrectionGUI:

    def __init__(self, image_path, detections, class_map, taxonomy=None, state_file=None,
                 window_size=(1360, 820), headless=False, max_window_size=None):
        if max_window_size is not None:          # backward compatibility
            window_size = max_window_size
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.img_h, self.img_w = self.image.shape[:2]
        self.state_file = state_file
        self.headless = headless

        self.win_w = max(900, int(window_size[0]))
        self.win_h = max(700, int(window_size[1]))
        self.view_w = self.win_w - PANEL_W
        self.view_h = self.win_h - STATUS_H

        # classes
        self.class_map = {int(k): str(v) for k, v in dict(class_map).items()}
        self.class_ids = sorted(self.class_map)
        self.colors = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(self.class_ids)}
        if taxonomy is None:
            try:
                taxonomy = Taxonomy.load()
            except Exception as e:                 # GUI still usable without subcategories
                print(f"⚠️  Taxonomy not loaded ({e}): subcategories disabled.")
        self.taxonomy = taxonomy

        # detections
        self._uid = 0
        self.dets = [self._new_det(d, added=False) for d in detections]
        self.deleted = []
        self.legacy_counts = None

        # interaction state
        self.mode = "modify"
        self.selected = None              # index into self.dets
        self.hover = None
        self.add_cls = self._default_add_class()
        self.add_sub = defaultdict(lambda: UNSPECIFIED)
        self.show_boxes, self.show_labels, self.show_help = True, False, False
        self.drag = None
        self.mouse = (0, 0)
        self._last_click = None
        self.message = ""
        self.undo_stack, self.redo_stack = [], []
        self.panel_hits = []
        self.dirty = True

        # view
        self._pyramid = {1.0: self.image}
        self._base_key, self._base = None, None
        self.fit_view()

        if state_file and os.path.exists(state_file):
            self.load_state_from_file(state_file)

        self._qt_reset_pending = False
        if not headless:
            self._create_window()

    def _create_window(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(WINDOW_NAME, 20, 20)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

    def _reset_qt_view(self):
        """The Qt backend zooms the whole window by itself on every wheel event, on
        top of our own zoom, and offers no API to undo it: recreate the window."""
        self._qt_reset_pending = False
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)
        self._create_window()
        self.dirty = True

    # ------------------------------------------------------------ detections
    def _new_det(self, d, added):
        if isinstance(d, dict):
            cls, xc, yc, w, h = d["cls"], d["xc"], d["yc"], d["w"], d["h"]
            conf, sub = d.get("conf", 1.0), d.get("sub") or UNSPECIFIED
        else:
            cls, xc, yc, w, h = d[:5]
            conf = d[5] if len(d) > 5 else 1.0
            sub = d[7] if len(d) > 7 and d[7] else UNSPECIFIED
        self._uid += 1
        det = {"uid": self._uid, "cls": int(float(cls)), "xc": float(xc), "yc": float(yc),
               "w": abs(float(w)), "h": abs(float(h)), "conf": float(conf if conf is not None else 1.0),
               "sub": sub, "added": bool(added), "orig": None}
        if not self._sub_valid(det["cls"], det["sub"]):
            det["sub"] = UNSPECIFIED
        if not added:
            det["orig"] = {k: det[k] for k in ("cls", "sub", "xc", "yc", "w", "h")}
        return det

    def _default_add_class(self):
        for cid in self.class_ids:
            if self.class_map[cid].lower() in ("ic", "ics"):
                return cid
        return self.class_ids[0] if self.class_ids else 0

    def _subs(self, cls):
        if self.taxonomy is None:
            return []
        return self.taxonomy.subcategories(self.class_map.get(cls, ""))

    def _sub_valid(self, cls, sub):
        return sub == UNSPECIFIED or any(s.key == sub for s in self._subs(cls))

    def _sub_label(self, cls, sub, short=False):
        for s in self._subs(cls):
            if s.key == sub:
                return s.short if short else s.label
        return "" if short else "Unspecified"

    # ------------------------------------------------------------- geometry
    def fit_view(self):
        self.zoom = min(self.view_w / self.img_w, self.view_h / self.img_h)
        self.fit_zoom = self.zoom
        self.off_x = (self.view_w - self.img_w * self.zoom) / 2
        self.off_y = (self.view_h - self.img_h * self.zoom) / 2
        self.dirty = True

    def _clamp_view(self):
        for axis in ("x", "y"):
            size = (self.img_w if axis == "x" else self.img_h) * self.zoom
            view = self.view_w if axis == "x" else self.view_h
            off = getattr(self, "off_" + axis)
            off = (view - size) / 2 if size <= view else min(0.0, max(view - size, off))
            setattr(self, "off_" + axis, off)

    def zoom_at(self, factor, cx, cy):
        new = min(MAX_ZOOM, max(self.fit_zoom * 0.5, self.zoom * factor))
        if new == self.zoom:
            return
        self.off_x = cx - (cx - self.off_x) * new / self.zoom
        self.off_y = cy - (cy - self.off_y) * new / self.zoom
        self.zoom = new
        self._clamp_view()
        self.dirty = True

    def pan(self, dx, dy):
        self.off_x += dx
        self.off_y += dy
        self._clamp_view()
        self.dirty = True

    def to_img(self, x, y):
        return (x - self.off_x) / self.zoom, (y - self.off_y) / self.zoom

    def det_rect(self, d):
        """Display rectangle (x1, y1, x2, y2) as floats."""
        x1 = (d["xc"] - d["w"] / 2) * self.img_w * self.zoom + self.off_x
        y1 = (d["yc"] - d["h"] / 2) * self.img_h * self.zoom + self.off_y
        return x1, y1, x1 + d["w"] * self.img_w * self.zoom, y1 + d["h"] * self.img_h * self.zoom

    def _set_rect(self, d, x1, y1, x2, y2):
        """Set a detection from a display rectangle, clipped to the image."""
        ax1, ay1 = self.to_img(min(x1, x2), min(y1, y2))
        ax2, ay2 = self.to_img(max(x1, x2), max(y1, y2))
        ax1, ax2 = max(0.0, ax1), min(float(self.img_w), ax2)
        ay1, ay2 = max(0.0, ay1), min(float(self.img_h), ay2)
        d["xc"] = (ax1 + ax2) / 2 / self.img_w
        d["yc"] = (ay1 + ay2) / 2 / self.img_h
        d["w"] = max(ax2 - ax1, 1.0) / self.img_w
        d["h"] = max(ay2 - ay1, 1.0) / self.img_h

    def hits_at(self, x, y, tol=3):
        """Indices of boxes under (x, y), smallest first."""
        hits = []
        for i, d in enumerate(self.dets):
            x1, y1, x2, y2 = self.det_rect(d)
            if x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol:
                hits.append(((x2 - x1) * (y2 - y1), i))
        return [i for _, i in sorted(hits)]

    def handle_at(self, x, y):
        """Handle of the selected box under (x, y): e.g. 'tl', 'r', or None."""
        if self.selected is None:
            return None
        x1, y1, x2, y2 = self.det_rect(self.dets[self.selected])
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        pts = {"tl": (x1, y1), "t": (xm, y1), "tr": (x2, y1), "r": (x2, ym),
               "br": (x2, y2), "b": (xm, y2), "bl": (x1, y2), "l": (x1, ym)}
        if x2 - x1 < 4 * HANDLE_R or y2 - y1 < 4 * HANDLE_R:     # small box: corners only
            pts = {k: v for k, v in pts.items() if len(k) == 2}
        for name, (hx, hy) in pts.items():
            if abs(x - hx) <= HANDLE_R + 2 and abs(y - hy) <= HANDLE_R + 2:
                return name
        return None

    # --------------------------------------------------------------- history
    def _snapshot(self):
        return (copy.deepcopy(self.dets), copy.deepcopy(self.deleted),
                self.dets[self.selected]["uid"] if self.selected is not None else None)

    def checkpoint(self):
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > HISTORY_MAX:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore(self, snap):
        self.dets, self.deleted, uid = copy.deepcopy(snap[0]), copy.deepcopy(snap[1]), snap[2]
        self.selected = next((i for i, d in enumerate(self.dets) if d["uid"] == uid), None)
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            self.message = "Nothing to undo"
            return
        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self.message = "Undo"
        self.autosave()

    def redo(self):
        if not self.redo_stack:
            self.message = "Nothing to redo"
            return
        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self.message = "Redo"
        self.autosave()

    # ---------------------------------------------------------------- edits
    def delete(self, idx):
        if idx is None or not 0 <= idx < len(self.dets):
            return
        self.checkpoint()
        d = self.dets.pop(idx)
        if not d["added"]:                       # deleting an own addition just cancels it
            self.deleted.append(d)
        if self.selected == idx:
            self.selected = None
        elif self.selected is not None and self.selected > idx:
            self.selected -= 1
        self.hover = None
        self.message = f"Deleted {self.class_map.get(d['cls'], d['cls'])}"
        self.dirty = True
        self.autosave()

    def set_class(self, cls):
        if cls not in self.class_map:
            return
        if self.mode == "add" or self.selected is None:
            self.add_cls = cls
        if self.selected is not None:
            d = self.dets[self.selected]
            if d["cls"] != cls:
                self.checkpoint()
                d["cls"] = cls
                if not self._sub_valid(cls, d["sub"]):
                    d["sub"] = UNSPECIFIED
                self.message = f"Class -> {self.class_map[cls]}"
                self.autosave()
        self.dirty = True

    def set_sub(self, sub):
        cls = self.dets[self.selected]["cls"] if self.selected is not None else self.add_cls
        if not self._sub_valid(cls, sub):
            return
        if self.mode == "add" or self.selected is None:
            self.add_sub[cls] = sub
        if self.selected is not None:
            d = self.dets[self.selected]
            if d["sub"] != sub:
                self.checkpoint()
                d["sub"] = sub
                self.message = f"Subcategory -> {self._sub_label(cls, sub)}"
                self.autosave()
        self.dirty = True

    def cycle_sub(self, step):
        cls = self.dets[self.selected]["cls"] if self.selected is not None else self.add_cls
        keys = [UNSPECIFIED] + [s.key for s in self._subs(cls) if s.key != UNSPECIFIED]
        if len(keys) < 2:
            return
        cur = self.dets[self.selected]["sub"] if self.selected is not None else self.add_sub[cls]
        i = keys.index(cur) if cur in keys else 0
        self.set_sub(keys[(i + step) % len(keys)])

    def add_box(self, x1, y1, x2, y2):
        self.checkpoint()
        d = self._new_det({"cls": self.add_cls, "xc": 0, "yc": 0, "w": 0, "h": 0, "conf": 1.0,
                           "sub": self.add_sub[self.add_cls]}, added=True)
        self._set_rect(d, x1, y1, x2, y2)
        self.dets.append(d)
        self.selected = len(self.dets) - 1
        self.message = f"Added {self.class_map[self.add_cls]} - pick its class/subcategory in the panel"
        self.dirty = True
        self.autosave()

    # --------------------------------------------------------------- review
    def review_list(self):
        model = [i for i, d in enumerate(self.dets) if not d["added"]]
        return sorted(model, key=lambda i: (self.dets[i]["conf"], self.dets[i]["uid"]))

    def review_step(self, step):
        order = self.review_list()
        if not order:
            return
        if self.selected in order:
            pos = (order.index(self.selected) + step) % len(order)
        else:
            pos = 0 if step > 0 else len(order) - 1
        self.selected = order[pos]
        self.center_on(self.selected)
        d = self.dets[self.selected]
        self.message = (f"Review {pos + 1}/{len(order)}: {self.class_map.get(d['cls'])} "
                        f"conf {d['conf']:.2f}")

    def center_on(self, idx):
        d = self.dets[idx]
        size = max(d["w"] * self.img_w, d["h"] * self.img_h) * self.zoom
        if size < 40:                                # make small parts visible
            self.zoom = min(MAX_ZOOM, self.zoom * 120 / max(size, 1))
        elif size > 0.8 * min(self.view_w, self.view_h):
            self.zoom = max(self.fit_zoom, 0.6 * min(self.view_w, self.view_h) /
                            max(d["w"] * self.img_w, d["h"] * self.img_h))
        self.off_x = self.view_w / 2 - d["xc"] * self.img_w * self.zoom
        self.off_y = self.view_h / 2 - d["yc"] * self.img_h * self.zoom
        self._clamp_view()
        self.dirty = True

    # ---------------------------------------------------------------- mouse
    def on_mouse(self, event, x, y, flags, param=None):
        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            self._on_wheel(event, x, y, flags)
            return
        self.mouse = (x, y)
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        # a drag in progress owns the mouse, wherever the pointer is
        if self.drag is not None:
            buttons = flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON | cv2.EVENT_FLAG_MBUTTON)
            if event == cv2.EVENT_MOUSEMOVE and buttons:
                self._drag_move(x, y)
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
                self._drag_end(x, y)
            elif event == cv2.EVENT_MOUSEMOVE or event in (
                    cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                # button released outside the window (no UP event received): close the drag
                self._drag_end(*self.drag["last"])
                if event != cv2.EVENT_MOUSEMOVE:
                    self.on_mouse(event, x, y, flags, param)
            return

        if event == cv2.EVENT_LBUTTONDOWN and x >= self.view_w:
            self._panel_click(x, y)
            return
        if x >= self.view_w or y >= self.view_h:
            if self.hover is not None:
                self.hover, self.dirty = None, True
            return

        if event == cv2.EVENT_MOUSEMOVE:
            hits = self.hits_at(x, y)
            hover = hits[0] if hits and self.mode in ("modify", "delete") else None
            if hover != self.hover:
                self.hover, self.dirty = hover, True
            return

        if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and shift):
            self._start("pan", x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._start("draw", x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._left_down(x, y)

    def _on_wheel(self, event, x, y, flags):
        """Wheel / trackpad scroll: vertical zooms around the pointer, horizontal pans.

        Backends differ: GTK, Qt and Windows put the delta in the high word of
        ``flags`` (120 per notch) and the pointer in (x, y); Cocoa (macOS) puts the
        scroll deltas in (x, y) themselves (about 10 per notch, small precise
        values for trackpads) and leaves the pointer position out.
        """
        if BACKEND == "COCOA":
            dx, dy, unit = x, y, 10.0
            px, py = self.mouse
        else:
            d = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags >> 16
            dx, dy = (d, 0) if event == cv2.EVENT_MOUSEHWHEEL else (0, d)
            unit = 120.0
            px, py = x, y
            self.mouse = (x, y)
            self._qt_reset_pending = BACKEND == "QT"
        if dy and px < self.view_w and py < self.view_h:
            steps = min(3.0, max(0.05, abs(dy) / unit))      # trackpads: many small steps
            self.zoom_at(ZOOM_STEP ** (steps if dy > 0 else -steps), px, py)
        if dx:
            self.pan(max(-120.0, min(120.0, -dx * 60.0 / unit)), 0)

    def _start(self, kind, x, y, **kw):
        self.drag = {"type": kind, "start": (x, y), "last": (x, y), "moved": False, **kw}
        self.dirty = True

    def _left_down(self, x, y):
        if self.mode == "add":
            self._start("draw", x, y)
            return
        hits = self.hits_at(x, y)
        if self.mode == "delete":
            if hits:
                self.delete(hits[0])
            else:
                self._start("pan", x, y)
            return
        # modify
        handle = self.handle_at(x, y)
        if handle:
            self._start("resize", x, y, handle=handle, rect=self.det_rect(self.dets[self.selected]))
            return
        if not hits:
            self._start("pan", x, y, deselect=True)
            return
        same_spot = self._last_click and math.dist(self._last_click, (x, y)) <= CLICK_TOL
        if self.selected in hits and same_spot and len(hits) > 1:
            self.selected = hits[(hits.index(self.selected) + 1) % len(hits)]
        elif self.selected not in hits:
            self.selected = hits[0]
        self._last_click = (x, y)
        self._start("move", x, y, rect=self.det_rect(self.dets[self.selected]))

    def _drag_move(self, x, y):
        dr = self.drag
        if not dr["moved"] and math.dist(dr["start"], (x, y)) > CLICK_TOL:
            dr["moved"] = True
            if dr["type"] in ("move", "resize"):
                self.checkpoint()
        if dr["type"] == "pan":
            self.pan(x - dr["last"][0], y - dr["last"][1])
        elif dr["moved"] and dr["type"] == "move":
            x1, y1, x2, y2 = dr["rect"]
            # keep the whole box inside the image (moving must not resize it)
            lo_x, hi_x = self.off_x, self.off_x + self.img_w * self.zoom
            lo_y, hi_y = self.off_y, self.off_y + self.img_h * self.zoom
            dx = min(max(x - dr["start"][0], lo_x - x1), hi_x - x2)
            dy = min(max(y - dr["start"][1], lo_y - y1), hi_y - y2)
            self._set_rect(self.dets[self.selected], x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        elif dr["moved"] and dr["type"] == "resize":
            x1, y1, x2, y2 = dr["rect"]
            h = dr["handle"]
            if "l" in h: x1 = x
            if "r" in h: x2 = x
            if "t" in h: y1 = y
            if "b" in h: y2 = y
            self._set_rect(self.dets[self.selected], x1, y1, x2, y2)
        dr["last"] = (x, y)
        self.dirty = True

    def _drag_end(self, x, y):
        dr, self.drag = self.drag, None
        self.dirty = True
        if dr["type"] == "draw":
            (sx, sy) = dr["start"]
            if abs(x - sx) >= MIN_BOX_PX and abs(y - sy) >= MIN_BOX_PX:
                self.add_box(sx, sy, x, y)
            else:                                   # a click in ADD mode selects
                hits = self.hits_at(sx, sy)
                self.selected = hits[0] if hits else None
        elif dr["type"] in ("move", "resize") and dr["moved"]:
            self.message = "Box moved" if dr["type"] == "move" else "Box resized"
            self.autosave()
        elif dr["type"] == "pan" and not dr["moved"] and dr.get("deselect"):
            self.selected = None

    # ---------------------------------------------------------------- panel
    def _panel_click(self, x, y):
        for (x1, y1, x2, y2, action, arg) in self.panel_hits:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._do(action, arg)
                self.dirty = True
                return

    def _do(self, action, arg=None):
        if action == "mode":
            self.mode = arg
            self.message = f"{arg.upper()} mode"
        elif action == "class":
            self.set_class(arg)
        elif action == "sub":
            self.set_sub(arg)
        elif action == "delete":
            self.delete(self.selected)
        elif action == "undo":
            self.undo()
        elif action == "redo":
            self.redo()
        elif action == "fit":
            self.fit_view()
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "review":
            self.review_step(arg)
        elif action == "save":
            self.save_state_to_file(self.state_file)
        self.dirty = True

    # ------------------------------------------------------------- keyboard
    def on_key(self, code):
        """Handle one key code from cv2.waitKeyEx. Returns False to quit."""
        if code is None or code < 0:
            return True
        self.dirty = True
        if code in K_LEFT:  self.pan(80, 0);  return True
        if code in K_RIGHT: self.pan(-80, 0); return True
        if code in K_UP:    self.pan(0, 80);  return True
        if code in K_DOWN:  self.pan(0, -80); return True
        if code in K_DELETE:
            self.delete(self.selected)
            return True
        if code > 255:
            code &= 0xFF
        if code == K_ESC:
            if self.drag is not None:
                self.drag = None
            elif self.show_help:
                self.show_help = False
            elif self.selected is not None:
                self.selected = None
            else:
                return False
            return True
        if code == K_CTRL_Z:
            self.undo(); return True
        if code == K_CTRL_Y:
            self.redo(); return True
        ch = chr(code).lower() if 32 <= code < 127 else ""
        if ch == "q":
            return False
        if ch in ("m", "c"):
            self._do("mode", "modify")
        elif ch == "a":
            self._do("mode", "add")
        elif ch == "d":
            self._do("mode", "delete")
        elif ch == "x":
            self.delete(self.selected)
        elif ch.isdigit() and ch != "0":
            i = int(ch) - 1
            if i < len(self.class_ids):
                self.set_class(self.class_ids[i])
        elif ch == "0":
            self.set_sub(UNSPECIFIED)
        elif ch == "]":
            self.cycle_sub(1)
        elif ch == "[":
            self.cycle_sub(-1)
        elif ch == "n":
            self.review_step(1)
        elif ch == "p":
            self.review_step(-1)
        elif ch == "f":
            self.fit_view()
        elif ch in ("+", "="):
            self.zoom_at(ZOOM_STEP ** 2, self.view_w / 2, self.view_h / 2)
        elif ch in ("-", "_"):
            self.zoom_at(1 / ZOOM_STEP ** 2, self.view_w / 2, self.view_h / 2)
        elif ch in ("z", "u"):
            self.undo()
        elif ch in ("y", "r"):
            self.redo()
        elif ch == "s":
            self.save_state_to_file(self.state_file)
        elif ch == "b":
            self.show_boxes = not self.show_boxes
        elif ch == "t":
            self.show_labels = not self.show_labels
        elif ch in ("h", "?"):
            self.show_help = not self.show_help
        return True

    # ------------------------------------------------------------- render
    def _level(self, scale):
        if scale not in self._pyramid:
            w = max(1, int(round(self.img_w * scale)))
            h = max(1, int(round(self.img_h * scale)))
            self._pyramid[scale] = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)
        return self._pyramid[scale]

    def _render_base(self):
        key = (round(self.zoom, 6), round(self.off_x, 1), round(self.off_y, 1), self.view_w, self.view_h)
        if key == self._base_key:
            return self._base
        base = np.full((self.view_h, self.view_w, 3), VIEW_BG, np.uint8)
        s = 1.0
        while s / 2 >= self.zoom and s > 1 / 16:     # coarsest level still >= zoom
            s /= 2
        lvl = self._level(s)
        lh, lw = lvl.shape[:2]
        # visible part of the image, in level pixels
        ix0, iy0 = self.to_img(0, 0)
        ix1, iy1 = self.to_img(self.view_w, self.view_h)
        lx0, ly0 = max(0, int(math.floor(ix0 * s))), max(0, int(math.floor(iy0 * s)))
        lx1, ly1 = min(lw, int(math.ceil(ix1 * s))), min(lh, int(math.ceil(iy1 * s)))
        if lx1 > lx0 and ly1 > ly0:
            f = self.zoom / s
            dx0, dy0 = lx0 * f + self.off_x, ly0 * f + self.off_y
            dw = max(1, int(round((lx1 - lx0) * f)))
            dh = max(1, int(round((ly1 - ly0) * f)))
            interp = cv2.INTER_NEAREST if f >= 3 else (cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
            patch = cv2.resize(lvl[ly0:ly1, lx0:lx1], (dw, dh), interpolation=interp)
            px, py = int(round(dx0)), int(round(dy0))
            sx, sy = max(0, -px), max(0, -py)
            ex, ey = min(dw, self.view_w - px), min(dh, self.view_h - py)
            if ex > sx and ey > sy:
                base[py + sy:py + ey, px + sx:px + ex] = patch[sy:ey, sx:ex]
        self._base_key, self._base = key, base
        return base

    def _draw_tag(self, img, text, x, y, color):
        tw, th = cv2.getTextSize(text, FONT, 0.42, 1)[0]
        y = max(th + 4, int(y))
        x = int(min(max(0, x), self.view_w - tw - 6))
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 2), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        _text(img, text, (x + 3, y - 2), 0.42, (0, 0, 0) if lum > 140 else (255, 255, 255))

    def _det_tag(self, d):
        name = self.class_map.get(d["cls"], str(d["cls"]))
        short = self._sub_label(d["cls"], d["sub"], short=True)
        tag = f"{name} {short}".strip()
        return _ascii(tag + ("" if d["added"] else f" {d['conf']:.2f}"))

    def _draw_view(self):
        img = self._render_base().copy()
        thin = 1 if self.zoom < 0.35 and len(self.dets) > 150 else 2
        if self.show_boxes:
            for i, d in enumerate(self.dets):
                x1, y1, x2, y2 = self.det_rect(d)
                if x2 < 0 or y2 < 0 or x1 > self.view_w or y1 > self.view_h or i == self.selected:
                    continue
                col = self.colors.get(d["cls"], (0, 255, 0))
                if i == self.hover:
                    col = DELETE_COLOR if self.mode == "delete" else SELECT_COLOR
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, col, thin + (1 if i == self.hover else 0))
                if d["added"]:
                    cv2.circle(img, (p2[0], p1[1]), 4, (255, 255, 255), -1)
                    cv2.circle(img, (p2[0], p1[1]), 3, col, -1)
                if d["sub"] != UNSPECIFIED:              # small filled corner = subcategory set
                    tri = np.array([p1, (p1[0] + 7, p1[1]), (p1[0], p1[1] + 7)], np.int32)
                    cv2.fillConvexPoly(img, tri, col)
                if self.show_labels or i == self.hover:
                    self._draw_tag(img, self._det_tag(d), x1, y1 - 2, col)
        if self.selected is not None:
            d = self.dets[self.selected]
            x1, y1, x2, y2 = (int(v) for v in self.det_rect(d))
            col = self.colors.get(d["cls"], (0, 255, 0))
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 4)
            cv2.rectangle(img, (x1, y1), (x2, y2), SELECT_COLOR, 2)
            small = (x2 - x1) < 4 * HANDLE_R or (y2 - y1) < 4 * HANDLE_R
            xm, ym = (x1 + x2) // 2, (y1 + y2) // 2
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if not small:
                pts += [(xm, y1), (x2, ym), (xm, y2), (x1, ym)]
            for (hx, hy) in pts:
                cv2.rectangle(img, (hx - HANDLE_R, hy - HANDLE_R), (hx + HANDLE_R, hy + HANDLE_R), (0, 0, 0), -1)
                cv2.rectangle(img, (hx - HANDLE_R + 1, hy - HANDLE_R + 1),
                              (hx + HANDLE_R - 1, hy + HANDLE_R - 1), SELECT_COLOR, -1)
            self._draw_tag(img, self._det_tag(d), x1, y1 - HANDLE_R - 2, col)
        if self.drag and self.drag["type"] == "draw":
            (sx, sy), (cx, cy) = self.drag["start"], self.drag["last"]
            col = self.colors.get(self.add_cls, (0, 255, 0))
            cv2.rectangle(img, (int(sx), int(sy)), (int(cx), int(cy)), col, 2)
        if self.show_help:
            self._draw_help(img)
        return img

    def _draw_help(self, img):
        h = 30 + 22 * len(HELP_LINES)
        w = min(self.view_w - 40, 700)
        x0, y0 = (self.view_w - w) // 2, max(10, (self.view_h - h) // 2)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
        for i, line in enumerate(HELP_LINES):
            _text(img, _fit_text(line, w - 30, 0.48), (x0 + 16, y0 + 30 + 22 * i), 0.48, (235, 235, 235))

    def _draw_status(self):
        bar = np.full((STATUS_H, self.view_w, 3), (28, 28, 28), np.uint8)
        hints = {"modify": "click: select   drag box: move   drag handle: resize   drag background: pan",
                 "add": f"drag: draw a {self.class_map.get(self.add_cls, '')}   click: select   Shift/right-drag: pan",
                 "delete": "click a box to delete it   drag background: pan"}
        left = f"{self.mode.upper()}  |  {hints[self.mode]}"
        _text(bar, _fit_text(left, self.view_w * 0.62, 0.45), (10, 19), 0.45, (230, 230, 230))
        msg = _fit_text(_ascii(self.message), self.view_w * 0.36, 0.45)
        _text(bar, msg, (self.view_w - _text_w(msg) - 10, 19), 0.45, (130, 210, 255))
        return bar

    def _button(self, panel, x1, y1, x2, y2, label, action, arg=None, active=False, scale=0.45):
        cv2.rectangle(panel, (x1, y1), (x2, y2), P_ACCENT if active else P_BTN, -1)
        tw = _text_w(label, scale, 2 if active else 1)
        _text(panel, label, ((x1 + x2 - tw) // 2, (y1 + y2) // 2 + 5), scale,
              (255, 255, 255) if active else P_TEXT, 2 if active else 1)
        self.panel_hits.append((x1 + self.view_w, y1, x2 + self.view_w, y2, action, arg))

    def _row(self, panel, y, rh, label, right, action, arg, active, swatch=None, radio=False):
        x1, x2 = 12, PANEL_W - 12
        if active:
            cv2.rectangle(panel, (x1 - 4, y), (x2 + 4, y + rh - 2), (250, 225, 190), -1)
        cx = x1 + 6
        if swatch is not None:
            cv2.rectangle(panel, (x1, y + rh // 2 - 6), (x1 + 12, y + rh // 2 + 5), swatch, -1)
        elif radio:
            cv2.circle(panel, (cx, y + rh // 2 - 1), 5, P_ACCENT if active else P_MUTED, 2 if not active else -1)
        rw = _text_w(right, 0.4) if right else 0
        _text(panel, _fit_text(_ascii(label), x2 - x1 - 26 - rw - 8, 0.44), (x1 + 20, y + rh - 6), 0.44,
              P_TEXT, 2 if active else 1)
        if right:
            _text(panel, right, (x2 - rw, y + rh - 6), 0.4, P_MUTED)
        self.panel_hits.append((self.view_w, y, self.win_w, y + rh - 1, action, arg))

    def _draw_panel(self):
        p = np.full((self.win_h, PANEL_W, 3), P_BG, np.uint8)
        cv2.line(p, (0, 0), (0, self.win_h), P_LINE, 1)
        self.panel_hits = []
        y = 26
        _text(p, "InventoryVision", (12, y), 0.62, P_TEXT, 2)
        _text(p, _fit_text(_ascii(os.path.basename(self.image_path)), PANEL_W - 24, 0.4), (12, y + 18), 0.4, P_MUTED)
        y += 32

        # modes
        bw = (PANEL_W - 24 - 12) // 3
        for i, (m, lab) in enumerate((("modify", "MODIFY (M)"), ("add", "ADD (A)"), ("delete", "DELETE (D)"))):
            x1 = 12 + i * (bw + 6)
            self._button(p, x1, y, x1 + bw, y + 30, lab, "mode", m, self.mode == m, 0.42)
        y += 42

        sel = self.dets[self.selected] if self.selected is not None else None
        target_cls = sel["cls"] if sel else self.add_cls
        subs = self._subs(target_cls)
        fixed = 470
        n_rows = len(self.class_ids) + max(1, len(subs))
        rh = int(min(21, max(15, (self.win_h - fixed) / max(1, n_rows))))

        # classes
        counts = defaultdict(int)
        for d in self.dets:
            counts[d["cls"]] += 1
        head = "CLASS of selection  (1-9)" if sel else "CLASS for new boxes  (1-9)"
        _text(p, head, (12, y + 12), 0.42, P_MUTED, 1)
        y += 18
        for i, cid in enumerate(self.class_ids):
            label = f"{i + 1}  {self.class_map[cid]}" if i < 9 else self.class_map[cid]
            self._row(p, y, rh, label, str(counts[cid]), "class", cid, cid == target_cls, swatch=self.colors[cid])
            y += rh
        y += 8

        # subcategories
        cls_name = self.class_map.get(target_cls, "")
        _text(p, _fit_text(f"SUBCATEGORY - {cls_name}  ([ ] 0)", PANEL_W - 24, 0.42), (12, y + 12), 0.42, P_MUTED)
        y += 18
        current = sel["sub"] if sel else self.add_sub[target_cls]
        if not subs:
            _text(p, "no subcategory for this class", (20, y + rh - 6), 0.42, P_MUTED)
            y += rh
        for s in subs:
            right = ""
            if s.dpr is not None:
                right = f"DPR {s.dpr.nominal:.0%}"
            self._row(p, y, rh, s.label, right, "sub", s.key, s.key == current, radio=True)
            y += rh
        y += 8
        cv2.line(p, (12, y), (PANEL_W - 12, y), P_LINE, 1)
        y += 6

        # selection
        if sel:
            _text(p, f"SELECTED  #{self.selected + 1}  " + ("(added)" if sel["added"] else f"conf {sel['conf']:.2f}"),
                  (12, y + 14), 0.44, P_TEXT, 1)
            wpx, hpx = sel["w"] * self.img_w, sel["h"] * self.img_h
            info = f"{wpx:.0f} x {hpx:.0f} px"
            if sel["orig"] and sel["orig"]["cls"] != sel["cls"]:
                info += f"   model said: {self.class_map.get(sel['orig']['cls'], '?')}"
            _text(p, _fit_text(_ascii(info), PANEL_W - 130, 0.4), (12, y + 32), 0.4, P_MUTED)
            self._button(p, PANEL_W - 110, y + 4, PANEL_W - 12, y + 34, "Delete (X)", "delete")
        else:
            _text(p, "Nothing selected", (12, y + 14), 0.44, P_MUTED)
            _text(p, "click a box, or press N to review", (12, y + 32), 0.4, P_MUTED)
        y += 44

        # review
        order = self.review_list()
        pos = order.index(self.selected) + 1 if self.selected in order else 0
        bw2 = (PANEL_W - 24 - 8) // 2
        self._button(p, 12, y, 12 + bw2, y + 28, "< Prev (P)", "review", -1)
        self._button(p, 20 + bw2, y, PANEL_W - 12, y + 28, "Next (N) >", "review", 1)
        _text(p, f"lowest confidence first   {pos}/{len(order)}", (12, y + 44), 0.4, P_MUTED)
        y += 54

        # corrections
        s = self.corrections_summary()
        _text(p, "CORRECTIONS", (12, y + 12), 0.42, P_MUTED)
        _text(p, f"{s['added']} added   {s['deleted']} deleted   {s['reclassified']} reclassified",
              (12, y + 30), 0.42, P_TEXT)
        _text(p, f"{s['subcategorized']} subcategories set   {s['geometry']} moved/resized",
              (12, y + 47), 0.42, P_TEXT)
        y += 56

        # footer
        bw4 = (PANEL_W - 24 - 18) // 4
        for i, (lab, act) in enumerate((("Undo", "undo"), ("Redo", "redo"), ("Fit", "fit"), ("Help", "help"))):
            x1 = 12 + i * (bw4 + 6)
            self._button(p, x1, self.win_h - 56, x1 + bw4, self.win_h - 30, lab, act, None,
                         act == "help" and self.show_help, 0.42)
        _text(p, f"zoom {self.zoom / self.fit_zoom:.1f}x fit   {len(self.dets)} boxes   S save   Q quit",
              (12, self.win_h - 12), 0.4, P_MUTED)
        return p

    def render(self):
        frame = np.empty((self.win_h, self.win_w, 3), np.uint8)
        frame[:self.view_h, :self.view_w] = self._draw_view()
        frame[self.view_h:, :self.view_w] = self._draw_status()
        frame[:, self.view_w:] = self._draw_panel()
        return frame

    # ------------------------------------------------------------ statistics
    def corrections_summary(self):
        added = [d for d in self.dets if d["added"]]
        model = [d for d in self.dets if not d["added"]]
        reclass = [d for d in model if d["cls"] != d["orig"]["cls"]]
        geom = [d for d in model if _box_changed(d, d["orig"])]
        subbed = [d for d in self.dets
                  if d["sub"] != (d["orig"]["sub"] if d["orig"] else UNSPECIFIED)]
        return {"added": len(added), "deleted": len(self.deleted), "reclassified": len(reclass),
                "geometry": len(geom), "subcategorized": len(subbed),
                "operations": len(added) + len(self.deleted) + len(reclass),
                "_added": added, "_reclass": reclass}

    def _per_class_counts(self):
        s = self.corrections_summary()
        name = lambda c: self.class_map.get(c, str(c))
        added, deleted = defaultdict(int), defaultdict(int)
        moves = defaultdict(lambda: defaultdict(int))
        for d in s["_added"]:
            added[name(d["cls"])] += 1
        for d in self.deleted:
            deleted[name(d["orig"]["cls"])] += 1
        for d in s["_reclass"]:
            moves[name(d["orig"]["cls"])][name(d["cls"])] += 1
        return dict(added), dict(deleted), {k: dict(v) for k, v in moves.items()}

    # ---------------------------------------------------------- persistence
    def state_dict(self):
        s = self.corrections_summary()
        added, deleted, moves = self._per_class_counts()
        clean = lambda d: {k: v for k, v in d.items()}
        return {
            "version": 2,
            "image": os.path.basename(self.image_path),
            "image_size": [self.img_w, self.img_h],
            "class_map": {str(k): v for k, v in self.class_map.items()},
            "detections": [clean(d) for d in self.dets],
            "deleted": [clean(d) for d in self.deleted],
            "view": {"zoom": self.zoom, "off_x": self.off_x, "off_y": self.off_y, "mode": self.mode},
            # summary, also read by component_to_mass (keys are class names)
            "added_count": s["added"], "deleted_count": s["deleted"],
            "reclassified_count": s["reclassified"], "geometry_count": s["geometry"],
            "subcategory_count": s["subcategorized"], "operations": s["operations"],
            "added_by_class": added, "deleted_by_class": deleted, "reclassified_from_to": moves,
        }

    def save_state_to_file(self, filepath, quiet=False):
        if not filepath:
            return
        try:
            tmp = f"{filepath}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state_dict(), f, indent=1)
            os.replace(tmp, filepath)
            self.message = "Saved"
            if not quiet:
                print(f"✅ Annotation state saved to: {filepath}")
        except OSError as e:
            self.message = "Save FAILED"
            print(f"❌ Error saving state to {filepath}: {e}")

    def autosave(self):
        if self.state_file:
            msg = self.message
            self.save_state_to_file(self.state_file, quiet=True)
            self.message = msg

    def load_state_from_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError) as e:
            print(f"❌ Error loading state from {filepath}: {e}")
            return
        size = state.get("image_size")
        if size and tuple(size) != (self.img_w, self.img_h):
            print(f"⚠️  {filepath} was made for a {size[0]}x{size[1]} image, this one is "
                  f"{self.img_w}x{self.img_h}: state ignored.")
            return
        raw = state.get("detections", [])
        if state.get("version", 1) >= 2:
            self.dets, self.deleted = [], []
            for d in raw:
                nd = self._new_det(d, added=d.get("added", False))
                if d.get("orig"):
                    nd["orig"] = dict(d["orig"])
                self.dets.append(nd)
            for d in state.get("deleted", []):
                nd = self._new_det(d, added=False)
                if d.get("orig"):
                    nd["orig"] = dict(d["orig"])
                self.deleted.append(nd)
        else:                                       # version 1: [cls, xc, yc, w, h, conf, is_added]
            self.dets = [self._new_det(d, added=bool(d[6]) if len(d) > 6 else False) for d in raw]
            self.deleted = []
            self.legacy_counts = {k: state.get(k, 0) for k in ("added_count", "deleted_count", "reclassified_count")}
            print("ℹ️  Version-1 state loaded: boxes kept, correction counters restart from here "
                  f"(previous session: {self.legacy_counts}).")
        view = state.get("view") or {}
        if "zoom" in view:
            self.zoom = float(view["zoom"])
            self.off_x, self.off_y = float(view.get("off_x", 0)), float(view.get("off_y", 0))
            self._clamp_view()
        self.mode = view.get("mode", "modify") if view.get("mode") in ("modify", "add", "delete") else "modify"
        self.selected = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = True
        print(f"✅ Annotation state loaded from: {filepath} ({len(self.dets)} boxes)")

    def result(self):
        return [{"cls": d["cls"], "xc": d["xc"], "yc": d["yc"], "w": d["w"], "h": d["h"],
                 "conf": d["conf"], "sub": d["sub"]} for d in self.dets]

    # ----------------------------------------------------------------- log
    def print_log(self):
        s = self.corrections_summary()
        added, deleted, moves = self._per_class_counts()
        print("\n" + "=" * 70)
        print("HUMAN CORRECTION LOG".center(70))
        print("=" * 70)
        print(f"  Added:            {s['added']:>5}")
        print(f"  Deleted:          {s['deleted']:>5}")
        print(f"  Reclassified:     {s['reclassified']:>5}")
        print(f"  Operations (add + delete + reclassify): {s['operations']}")
        print(f"  Moved / resized:  {s['geometry']:>5}")
        print(f"  Subcategories set:{s['subcategorized']:>5}")
        for cname in sorted(set(added) | set(deleted) | set(moves)):
            parts = []
            if added.get(cname): parts.append(f"+{added[cname]}")
            if deleted.get(cname): parts.append(f"-{deleted[cname]}")
            for to, n in (moves.get(cname) or {}).items():
                parts.append(f"{n} -> {to}")
            print(f"    {cname:15s} {'  '.join(parts)}")
        print("=" * 70 + "\n")

    # ----------------------------------------------------------------- run
    def run(self):
        if self.headless:
            raise RuntimeError("run() needs a window; create the GUI with headless=False")
        print("\n" + "=" * 70)
        print("HUMAN REVIEW - press H in the window for help")
        print("  M modify | A add | D delete | N/P review | Z undo | S save | Q quit")
        print("=" * 70 + "\n")
        self.message = "Press H for help"
        while True:
            if self._qt_reset_pending and self.drag is None:
                self._reset_qt_view()
            if self.dirty:
                cv2.imshow(WINDOW_NAME, self.render())
                self.dirty = False
            code = cv2.waitKeyEx(15)
            if code != -1 and not self.on_key(code):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        self.save_state_to_file(self.state_file)
        try:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self.print_log()
        return self.result()
