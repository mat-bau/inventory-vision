"""End-to-end run of main.py with a stubbed detector (no torch/ultralytics needed)."""

import builtins
import os
import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import MODEL_CLASSES

from scripts.labels import read_labels, write_labels


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    fake = types.ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = dict(MODEL_CLASSES)
    fake.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.chdir(tmp_path)

    import main
    from scripts import detection_classification

    def run(dets, answers, extra_args=()):
        img = np.full((800, 1000, 3), (40, 110, 30), np.uint8)
        cv2.imwrite("board.png", img)

        class FakeDetector:
            def __init__(self, model_path, bbox_reduction=0.0):
                pass

            def run(self, source, output_dir, conf_thresh=None, annotate=True):
                os.makedirs(f"{output_dir}/labels", exist_ok=True)
                write_labels(f"{output_dir}/labels/board.txt", dets)

        monkeypatch.setattr(detection_classification, "PCBDetector", FakeDetector)
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda *a: next(it))
        monkeypatch.setattr(sys, "argv", ["main.py", "--input", "board.png", "--model", "x.pt",
                                          "--output-xlsx", "out.xlsx", *extra_args])
        main.main()
        return tmp_path / "out.xlsx"
    return run


CROP = ["terminal", "0 0", "999 0", "999 799", "0 799", "80", "100"]   # board 100 x 80 mm


def test_subcategories_flow_to_excel(run_main, tmp_path):
    dets = [
        {"cls": 6, "xc": .2, "yc": .2, "w": .1, "h": .125, "conf": .95, "sub": "bga"},   # 10 x 10 mm
        {"cls": 6, "xc": .6, "yc": .6, "w": .05, "h": .0625, "conf": .9},                 # 5 x 5 mm
        {"cls": 0, "xc": .4, "yc": .4, "w": .01, "h": .01, "conf": .8, "sub": "smd_chip"},
        {"cls": 1, "xc": .5, "yc": .3, "w": .02, "h": .02, "conf": .8, "sub": "alu_tht_large"},
    ]
    out = run_main(dets, CROP, ["--bbox-uncertainty", "5"])

    assert pd.ExcelFile(out).sheet_names == [
        "Component Metrics", "LCI by subcategory", "Components", "Impacts", "Assumptions"]

    legacy = pd.read_excel(out, sheet_name="Component Metrics")
    ic_row = legacy[legacy["name"] == "market for integrated circuit, logic type"]
    assert ic_row["value"].iloc[0] == pytest.approx(125.0, rel=1e-3)          # original output kept

    lci = pd.read_excel(out, sheet_name="LCI by subcategory").set_index("sub_key")
    assert lci.loc["bga", "die_mm2_nom"] == pytest.approx(100 * 0.36, rel=1e-3)
    ic_unspec = lci[(lci["macro"] == "IC")].loc["unspecified"]
    assert ic_unspec["die_mm2_nom"] == pytest.approx(25 * 0.078, rel=1e-3)
    assert lci.loc["alu_tht_large", "ecoinvent_flow"].endswith("> 2cm height")

    cropped = read_labels(tmp_path / "output" / "labels" / "board_cropped.txt")
    assert [d["sub"] for d in cropped] == ["bga", "unspecified", "smd_chip", "alu_tht_large"]

    impacts = pd.read_excel(out, sheet_name="Impacts")
    assert (impacts["category"] == "MISSING").any()                          # template has no factors
