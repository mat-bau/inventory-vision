"""Interaction tests for the review GUI, without opening any window."""

import json

import cv2
import numpy as np
import pytest
from conftest import MODEL_CLASSES

from scripts.human_correction import HumanCorrectionGUI, load_state_detections
from scripts.taxonomy import UNSPECIFIED

L_DOWN, L_UP, MOVE = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE


@pytest.fixture
def image(tmp_path):
    img = np.full((1200, 2000, 3), (40, 110, 30), np.uint8)
    p = tmp_path / "board.png"
    cv2.imwrite(str(p), img)
    return str(p)


def dets():
    return [
        {"cls": 6, "xc": .30, "yc": .30, "w": .20, "h": .20, "conf": .95},   # big IC
        {"cls": 0, "xc": .30, "yc": .30, "w": .02, "h": .02, "conf": .40},   # resistor on top of it
        {"cls": 1, "xc": .70, "yc": .70, "w": .05, "h": .03, "conf": .25},   # lowest confidence
    ]


def make(image, tmp_path=None, **kw):
    state = str(tmp_path / "state.json") if tmp_path else None
    return HumanCorrectionGUI(image, dets(), MODEL_CLASSES, state_file=state, headless=True, **kw)


def center(g, i):
    x1, y1, x2, y2 = g.det_rect(g.dets[i])
    return (x1 + x2) / 2, (y1 + y2) / 2


def click(g, x, y, flags=0):
    g.on_mouse(L_DOWN, int(x), int(y), flags)
    g.on_mouse(L_UP, int(x), int(y), flags)


def drag(g, a, b, flags=0, steps=5):
    """Left-button drag; moves carry EVENT_FLAG_LBUTTON like every highgui backend does."""
    g.on_mouse(L_DOWN, int(a[0]), int(a[1]), flags | cv2.EVENT_FLAG_LBUTTON)
    for k in range(1, steps + 1):
        g.on_mouse(MOVE, int(a[0] + (b[0] - a[0]) * k / steps), int(a[1] + (b[1] - a[1]) * k / steps),
                   flags | cv2.EVENT_FLAG_LBUTTON)
    g.on_mouse(L_UP, int(b[0]), int(b[1]), flags)


def panel_hit(g, action, arg=None):
    g.render()
    for (x1, y1, x2, y2, a, v) in g.panel_hits:
        if a == action and (arg is None or v == arg):
            return (x1 + x2) / 2, (y1 + y2) / 2
    raise AssertionError(f"no panel control {action}={arg}")


def test_render_shape_and_fit(image):
    g = make(image)
    frame = g.render()
    assert frame.shape == (g.win_h, g.win_w, 3)
    assert g.zoom == pytest.approx(min(g.view_w / 2000, g.view_h / 1200))


def test_zoom_out_after_zoom_in_returns_to_fit(image):
    """Original GUI clamped zoom to >= 0.5, so a large image could never be fitted again."""
    g = make(image)
    fit = g.zoom
    for _ in range(10):
        g.on_mouse(cv2.EVENT_MOUSEWHEEL, 300, 300, 120 << 16)
    assert g.zoom > fit * 3
    for _ in range(30):
        g.on_mouse(cv2.EVENT_MOUSEWHEEL, 300, 300, -(120 << 16))
    assert g.zoom == pytest.approx(fit * 0.5)
    g.on_key(ord("f"))
    assert g.zoom == pytest.approx(fit)


def test_click_selects_smallest_then_cycles(image):
    g = make(image)
    x, y = center(g, 1)
    click(g, x, y)
    assert g.selected == 1                      # resistor, not the IC underneath
    click(g, x, y)
    assert g.selected == 0                      # second click on the same spot: next box
    click(g, 5, 5)                              # background click deselects
    assert g.selected is None


def test_modify_class_and_subcategory_with_undo_redo(image, tmp_path):
    g = make(image, tmp_path)
    x1, y1, _, _ = g.det_rect(g.dets[0])
    click(g, x1 + 10, y1 + 10)                  # on the IC, away from the resistor on top of it
    assert g.selected == 0
    click(g, *panel_hit(g, "sub", "bga"))
    assert g.dets[0]["sub"] == "bga"
    g.on_key(ord("]"))                          # next subcategory
    assert g.dets[0]["sub"] == "bga_memory"
    g.on_key(ord("2"))                          # class 2 = Capacitor -> IC subcategory reset
    assert g.dets[0]["cls"] == 1 and g.dets[0]["sub"] == UNSPECIFIED
    g.on_key(ord("z"))
    assert g.dets[0]["cls"] == 6 and g.dets[0]["sub"] == "bga_memory"
    g.on_key(26)                                # Ctrl-Z
    g.on_key(ord("y"))                          # redo
    assert g.dets[0]["sub"] == "bga_memory"
    s = g.corrections_summary()
    assert s["reclassified"] == 0 and s["subcategorized"] == 1
    state = json.loads((tmp_path / "state.json").read_text())   # autosaved
    assert state["version"] == 2 and state["detections"][0]["sub"] == "bga_memory"


def test_move_keeps_size_and_resize_changes_it(image):
    g = make(image)
    x, y = center(g, 2)
    click(g, x, y)
    w0, h0 = g.dets[2]["w"], g.dets[2]["h"]
    drag(g, (x, y), (x - 40, y - 30))
    assert g.dets[2]["w"] == pytest.approx(w0) and g.dets[2]["h"] == pytest.approx(h0)
    assert g.dets[2]["xc"] < .70
    x1, y1, x2, y2 = g.det_rect(g.dets[2])
    drag(g, (x2, y2), (x2 + 30, y2 + 20))       # bottom-right handle
    assert g.dets[2]["w"] > w0 and g.dets[2]["h"] > h0
    assert g.corrections_summary()["geometry"] == 1
    g.on_key(ord("z")); g.on_key(ord("z"))
    assert g.dets[2]["xc"] == pytest.approx(.70) and g.dets[2]["w"] == pytest.approx(w0)


def test_move_is_clamped_to_image(image):
    g = make(image)
    x, y = center(g, 2)
    click(g, x, y)
    w0 = g.dets[2]["w"]
    drag(g, (x, y), (x + 3000, y))
    assert g.dets[2]["w"] == pytest.approx(w0)
    assert g.dets[2]["xc"] + g.dets[2]["w"] / 2 == pytest.approx(1.0, abs=1e-3)


def test_add_mode_draws_selects_and_sets_class(image):
    g = make(image)
    g.on_key(ord("a"))
    click(g, *panel_hit(g, "class", 4))         # LED for new boxes
    drag(g, (600, 500), (650, 540))
    assert len(g.dets) == 4 and g.selected == 3
    new = g.dets[3]
    assert new["added"] and new["cls"] == 4 and new["conf"] == 1.0
    click(g, *panel_hit(g, "sub", "tht"))       # subcategory applied to the new box...
    assert new["sub"] == "tht"
    drag(g, (700, 500), (740, 530))             # ...and kept for the next ones
    assert g.dets[4]["cls"] == 4 and g.dets[4]["sub"] == "tht"
    click(g, *center(g, 2))                     # a simple click in ADD mode selects
    assert g.selected == 2 and len(g.dets) == 5


def test_tiny_drag_does_not_add(image):
    g = make(image)
    g.on_key(ord("a"))
    drag(g, (600, 500), (602, 501))
    assert len(g.dets) == 3


def test_delete_counts(image):
    g = make(image)
    g.on_key(ord("a"))
    drag(g, (600, 500), (650, 540))
    g.on_key(ord("x"))                          # deleting an own addition cancels it
    s = g.corrections_summary()
    assert s["added"] == 0 and s["deleted"] == 0
    g.on_key(ord("d"))
    click(g, *center(g, 1))
    s = g.corrections_summary()
    assert s["deleted"] == 1 and len(g.dets) == 2 and s["operations"] == 1


def test_selection_index_survives_deleting_another_box(image):
    g = make(image)
    click(g, *center(g, 2))
    g.delete(0)
    assert g.selected == 1 and g.dets[g.selected]["cls"] == 1


def test_review_goes_by_confidence(image):
    g = make(image)
    g.on_key(ord("n"))
    assert g.dets[g.selected]["conf"] == pytest.approx(.25)
    g.on_key(ord("n"))
    assert g.dets[g.selected]["conf"] == pytest.approx(.40)
    g.on_key(ord("p"))
    assert g.dets[g.selected]["conf"] == pytest.approx(.25)


def test_pan_with_shift_drag_in_add_mode(image):
    g = make(image)
    g.on_key(ord("a"))
    for _ in range(8):
        g.on_mouse(cv2.EVENT_MOUSEWHEEL, 400, 400, 120 << 16)
    ox = g.off_x
    drag(g, (400, 400), (300, 400), flags=cv2.EVENT_FLAG_SHIFTKEY)
    assert g.off_x == pytest.approx(ox - 100) and len(g.dets) == 3


def test_escape_deselects_then_quits(image):
    g = make(image)
    click(g, *center(g, 2))
    assert g.on_key(27) is True and g.selected is None
    assert g.on_key(27) is False


def test_state_roundtrip_and_history(image, tmp_path):
    g = make(image, tmp_path)
    click(g, *center(g, 1))
    g.on_key(ord("7"))                          # resistor -> IC
    g.on_key(ord("d"))
    click(g, *center(g, 2))                     # delete the capacitor
    g.on_key(ord("a"))
    drag(g, (600, 500), (650, 540))
    g.save_state_to_file(g.state_file, quiet=True)

    g2 = make(image, tmp_path)                  # reload
    s = g2.corrections_summary()
    assert (s["added"], s["deleted"], s["reclassified"]) == (1, 1, 1)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["reclassified_from_to"] == {"Resistor": {"IC": 1}}
    assert state["deleted_by_class"] == {"Capacitor": 1}
    assert len(load_state_detections(g2.state_file)) == 3


def test_version1_state_still_loads(image, tmp_path):
    v1 = {"detections": [[6, .3, .3, .2, .2, .9, False], [1, .5, .5, .05, .05, 1.0, True]],
          "zoom_level": 0.5, "pan_x": 0, "pan_y": 0, "added_count": 1, "deleted_count": 3,
          "reclassified_count": 2, "added_by_class": {"1": 1}}
    (tmp_path / "state.json").write_text(json.dumps(v1))
    g = make(image, tmp_path)
    assert len(g.dets) == 2 and g.dets[1]["added"] and g.legacy_counts["deleted_count"] == 3
    assert len(load_state_detections(str(tmp_path / "state.json"))) == 2


def test_state_for_other_image_is_ignored(image, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"version": 2, "image_size": [10, 10], "detections": []}))
    g = make(image, tmp_path)
    assert len(g.dets) == 3


def test_works_without_taxonomy_entries(image):
    g = HumanCorrectionGUI(image, dets(), {0: "Resistor", 1: "Capacitor", 6: "Widget"}, headless=True)
    click(g, *center(g, 0))
    g.render()
    assert g._subs(6) == []


def test_drag_closed_when_button_released_outside_window(image):
    g = make(image)
    x, y = center(g, 2)
    click(g, x, y)
    g.on_mouse(L_DOWN, int(x), int(y), cv2.EVENT_FLAG_LBUTTON)
    g.on_mouse(MOVE, int(x) - 30, int(y), cv2.EVENT_FLAG_LBUTTON)
    assert g.drag is not None
    g.on_mouse(MOVE, int(x) - 60, int(y), 0)        # moved back in, button already up
    assert g.drag is None
    assert g.dets[2]["xc"] < .70                    # the move done so far is kept
    g.on_key(ord("z"))
    assert g.dets[2]["xc"] == pytest.approx(.70)


def test_cocoa_wheel_uses_xy_as_deltas(image, monkeypatch):
    """macOS: OpenCV passes scroll deltas in (x, y) and nothing in flags."""
    import scripts.human_correction as hc
    monkeypatch.setattr(hc, "BACKEND", "COCOA")
    g = make(image)
    g.on_mouse(MOVE, 300, 300, 0)                   # pointer position comes from moves
    fit = g.zoom
    g.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 10, 0)      # one notch up
    assert g.zoom == pytest.approx(fit * 1.15)
    for _ in range(20):
        g.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 1, 0)   # trackpad: small precise deltas
    assert fit * 1.15 < g.zoom < fit * 1.15 * 1.4
    g.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, -10, 0)
    assert g.zoom < fit * 1.15 * 1.4 / 1.1
    ox = g.off_x
    g.on_mouse(cv2.EVENT_MOUSEHWHEEL, 10, 0, 0)     # horizontal scroll pans
    assert g.off_x != ox
    assert g.mouse == (300, 300)


def test_qt_wheel_requests_view_reset(image, monkeypatch):
    import scripts.human_correction as hc
    monkeypatch.setattr(hc, "BACKEND", "QT")
    g = make(image)
    g.on_mouse(cv2.EVENT_MOUSEWHEEL, 300, 300, 120 << 16)
    assert g._qt_reset_pending
