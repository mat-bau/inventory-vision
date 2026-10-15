"""
Read/write YOLO label files extended with an optional subcategory column.

Line format (one detection per line, space separated):

    cls xc yc w h [conf [subcategory]]

* ``cls`` integer class id, ``xc yc w h`` normalised box (YOLO format)
* ``conf`` detection confidence (1.0 for boxes drawn by a human)
* ``subcategory`` key from config/taxonomy.yaml, written only when it is not
  "unspecified" so that files stay readable by any YOLO tool.
"""

from __future__ import annotations

from scripts.taxonomy import UNSPECIFIED


def parse_label_line(line: str) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        det = {
            "cls": int(float(parts[0])),
            "xc": float(parts[1]),
            "yc": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
            "conf": float(parts[5]) if len(parts) > 5 else 1.0,
            "sub": UNSPECIFIED,
        }
    except ValueError:
        return None
    if len(parts) > 6 and parts[6] not in ("", "-"):
        det["sub"] = parts[6]
    return det


def format_label_line(cls, xc, yc, w, h, conf=1.0, sub=None) -> str:
    line = f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {float(conf if conf is not None else 1.0):.6f}"
    if sub and sub != UNSPECIFIED:
        line += f" {sub}"
    return line


def read_labels(path) -> list:
    dets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            det = parse_label_line(line)
            if det is not None:
                dets.append(det)
    return dets


def write_labels(path, detections) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(format_label_line(d["cls"], d["xc"], d["yc"], d["w"], d["h"],
                                      d.get("conf", 1.0), d.get("sub")) + "\n" for d in detections)
