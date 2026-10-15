import math

import pytest
import yaml
from conftest import MODEL_CLASSES, ROOT

from scripts import component_to_mass as c2m
from scripts.labels import (
    format_label_line,
    parse_label_line,
    read_labels,
    write_labels,
)
from scripts.lci import (
    ImpactFactors,
    aggregate,
    records_from_labels,
    write_excel_sheets,
)
from scripts.taxonomy import UNSPECIFIED, Taxonomy
from scripts.transform_detections import transform_yolo_detections


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.load(ROOT / "config" / "taxonomy.yaml")


# ----------------------------------------------------------------- taxonomy
def test_every_model_class_has_a_taxonomy_entry(tax):
    for name in MODEL_CLASSES.values():
        assert tax.macro(name) is not None, name


def test_unspecified_first_and_reproduces_original_flows(tax):
    original = {  # component_to_flow in main.py (original InventoryVision)
        "Resistor": "market for resistor, surface-mounted",
        "Capacitor": "market for capacitor, for surface-mounting",
        "Inductor": "market for inductor, low value multilayer chip",
        "Diode": "market for diode, glass-, for surface-mounting",
        "LED": "market for light emitting diode",
        "Transistor": "market for transistor, surface-mounted",
        "Connector": "market for electric connector, peripheral type buss",
        "IC": "market for integrated circuit, logic type",
    }
    for name, flow in original.items():
        subs = tax.subcategories(name)
        assert subs[0].key == UNSPECIFIED
        assert subs[0].ecoinvent_flow == flow
        assert subs[0].mass_model is None          # inherits the class regression


def test_ic_dpr_values(tax):
    assert tax.sub("IC", UNSPECIFIED).dpr.nominal == pytest.approx(0.078)   # Teehan, as in the paper
    bga = tax.sub("IC", "bga").dpr
    assert (bga.nominal, bga.low, bga.high) == (0.36, 0.07, 0.94)
    qfn = tax.sub("IC", "qfn").dpr
    assert qfn.range_reported is False and qfn.low == qfn.high == 0.28
    assert tax.sub("IC", "pop").stack_count == 2


def test_subcategory_inherits_flow_when_absent(tax):
    assert tax.sub("Capacitor", "mlcc").ecoinvent_flow == "market for capacitor, for surface-mounting"
    assert tax.sub("Capacitor", "alu_tht_large").ecoinvent_flow.endswith("> 2cm height")


def test_invalid_dpr_rejected():
    bad = {"classes": {"ics": {"dpr": {"nominal": 0.5, "low": 0.6, "high": 0.9}}}}
    with pytest.raises(ValueError):
        Taxonomy.from_dict(bad)


# ------------------------------------------------------------------- labels
def test_label_roundtrip(tmp_path):
    p = tmp_path / "l.txt"
    dets = [{"cls": 6, "xc": .5, "yc": .5, "w": .1, "h": .2, "conf": .87, "sub": "bga"},
            {"cls": 0, "xc": .1, "yc": .2, "w": .01, "h": .02, "conf": .5, "sub": UNSPECIFIED}]
    write_labels(p, dets)
    lines = p.read_text().splitlines()
    assert lines[0].endswith(" bga") and len(lines[1].split()) == 6   # unspecified not written
    back = read_labels(p)
    assert back[0]["sub"] == "bga" and back[1]["sub"] == UNSPECIFIED
    assert back[0]["conf"] == pytest.approx(.87)


def test_parse_legacy_line_without_confidence():
    d = parse_label_line("3 0.5 0.5 0.1 0.1")
    assert d["conf"] == 1.0 and d["sub"] == UNSPECIFIED
    assert parse_label_line("garbage") is None


def test_transform_keeps_subcategory(tmp_path):
    src, dst = tmp_path / "in.txt", tmp_path / "out.txt"
    src.write_text(format_label_line(6, .5, .5, .1, .1, .9, "qfn") + "\n" +
                   format_label_line(0, .4, .4, .05, .05, .8) + "\n")
    n = transform_yolo_detections(str(src), str(dst), [(0, 0), (999, 0), (999, 799), (0, 799)], 1000, 800)
    out = read_labels(dst)
    assert n == 2 and out[0]["sub"] == "qfn" and out[1]["sub"] == UNSPECIFIED


# ---------------------------------------------------------------------- LCI
def _write(tmp_path, dets):
    p = tmp_path / "cropped.txt"
    write_labels(p, dets)
    return p


def test_unspecified_mass_identical_to_component_to_mass(tmp_path, tax, capsys):
    dets = [{"cls": 0, "xc": .2, "yc": .2, "w": .02, "h": .01, "conf": .9},
            {"cls": 1, "xc": .3, "yc": .3, "w": .03, "h": .02, "conf": .9},
            {"cls": 7, "xc": .6, "yc": .6, "w": .08, "h": .05, "conf": .9},
            {"cls": 6, "xc": .5, "yc": .5, "w": .1, "h": .1, "conf": .9}]
    p = _write(tmp_path, dets)
    total_mass, counts, ic_area, unc, _ = c2m.calculate_mass_per_class(
        str(p), MODEL_CLASSES, 100.0, 80.0, bbox_uncertainty_pct=5.0)
    recs = records_from_labels(p, MODEL_CLASSES, tax, 100.0, 80.0, 5.0, "img")
    rows, _, _ = aggregate(recs, tax, ImpactFactors({}))
    by = {(r["macro"], r["sub_key"]): r for r in rows}
    for macro, key in (("Resistor", "resistors"), ("Capacitor", "capacitors"), ("Connector", "connectors")):
        r = by[(macro, UNSPECIFIED)]
        assert r["mass_mg_nom"] == pytest.approx(total_mass[key])
        assert r["mass_mg_low"] == pytest.approx(unc["per_class"][key]["combined_low"])
        assert r["mass_mg_high"] == pytest.approx(unc["per_class"][key]["combined_high"])
    ic = by[("IC", UNSPECIFIED)]
    assert ic["area_mm2"] == pytest.approx(ic_area)
    assert ic["die_mm2_nom"] == pytest.approx(ic_area * 0.078)


def test_die_area_by_package_and_stack(tmp_path, tax):
    # 10 mm x 10 mm packages on a 100 x 100 mm board
    dets = [{"cls": 6, "xc": .2, "yc": .2, "w": .1, "h": .1, "conf": .9, "sub": "bga"},
            {"cls": 6, "xc": .5, "yc": .5, "w": .1, "h": .1, "conf": .9, "sub": "pop"},
            {"cls": 6, "xc": .8, "yc": .8, "w": .1, "h": .1, "conf": .9, "sub": "qfn"}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 100, 100, 0.0, "img")
    assert [r["die_mm2_nom"] for r in recs] == pytest.approx([36.0, 72.0, 28.0])
    assert (recs[0]["die_mm2_low"], recs[0]["die_mm2_high"]) == pytest.approx((7.0, 94.0))
    assert recs[2]["die_mm2_low"] == recs[2]["die_mm2_high"] == pytest.approx(28.0)


def test_unknown_subcategory_falls_back(tmp_path, tax):
    warnings = []
    dets = [{"cls": 0, "xc": .5, "yc": .5, "w": .1, "h": .1, "conf": .9, "sub": "nope"}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 10, 10, 0, "img", warn=warnings.append)
    assert recs[0]["sub"] == UNSPECIFIED and warnings


def test_switch_has_no_mass_and_is_reported_missing(tmp_path, tax):
    dets = [{"cls": 8, "xc": .5, "yc": .5, "w": .1, "h": .1, "conf": .9}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 10, 10, 0, "img")
    assert recs[0]["kind"] == "none"
    fac = ImpactFactors.load(ROOT / "config" / "impact_factors.yaml")
    rows, totals, missing = aggregate(recs, tax, fac)
    assert any("no ecoinvent flow" in m for m in missing)


def test_impacts_with_factors(tmp_path, tax):
    dets = [{"cls": 0, "xc": .5, "yc": .5, "w": .02, "h": .01, "conf": .9},
            {"cls": 6, "xc": .2, "yc": .2, "w": .1, "h": .1, "conf": .9, "sub": "bga"}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 100, 100, 0, "img")
    raw = yaml.safe_load((ROOT / "config" / "impact_factors.yaml").read_text())
    raw["flows"]["market for resistor, surface-mounted"]["gwp"] = 100.0        # fake numbers
    raw["flows"]["market for integrated circuit, logic type"]["gwp"] = 1000.0
    raw["ic_die_conversion"]["market for integrated circuit, logic type"]["kg_per_cm2_die"] = 0.01
    fac = ImpactFactors(raw)
    rows, totals, missing = aggregate(recs, tax, fac)
    res = next(r for r in rows if r["macro"] == "Resistor")
    ic = next(r for r in rows if r["sub_key"] == "bga")
    assert res["gwp_nom"] == pytest.approx(res["mass_mg_nom"] * 1e-6 * 100.0)
    assert ic["ref_qty_nom"] == pytest.approx(36.0 / 100 * 0.01)                 # 36 mm2 die -> kg
    assert ic["gwp_nom"] == pytest.approx(36.0 / 100 * 0.01 * 1000.0)
    assert totals["gwp"]["nom"] == pytest.approx(res["gwp_nom"] + ic["gwp_nom"])
    assert totals["gwp"]["covered_rows"] == 2 and totals["mrd"]["covered_rows"] == 0


def test_die_cm2_basis(tmp_path, tax):
    dets = [{"cls": 6, "xc": .2, "yc": .2, "w": .1, "h": .1, "conf": .9, "sub": "csp"}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 100, 100, 0, "img")
    fac = ImpactFactors({"categories": {"gwp": {"unit": "kg"}},
                         "flows": {"market for integrated circuit, logic type": {"basis": "die_cm2", "gwp": 2.0}}})
    rows, totals, _ = aggregate(recs, tax, fac)
    assert rows[0]["ref_unit"] == "cm2 die"
    assert totals["gwp"]["nom"] == pytest.approx(72.5 / 100 * 2.0)


def test_excel_sheets(tmp_path, tax):
    import pandas as pd
    dets = [{"cls": 6, "xc": .2, "yc": .2, "w": .1, "h": .1, "conf": .9, "sub": "bga"},
            {"cls": 1, "xc": .5, "yc": .5, "w": .02, "h": .01, "conf": .7, "sub": "mlcc"}]
    recs = records_from_labels(_write(tmp_path, dets), MODEL_CLASSES, tax, 100, 100, 0, "img")
    out = tmp_path / "o.xlsx"
    with pd.ExcelWriter(out) as w:
        pd.DataFrame([{"a": 1}]).to_excel(w, sheet_name="Component Metrics", index=False)
        write_excel_sheets(w, recs, tax, ImpactFactors.load(ROOT / "config" / "impact_factors.yaml"))
    sheets = pd.ExcelFile(out).sheet_names
    assert sheets == ["Component Metrics", "LCI by subcategory", "Components", "Impacts", "Assumptions"]
    lci = pd.read_excel(out, sheet_name="LCI by subcategory")
    assert set(lci["sub_key"]) == {"bga", "mlcc"}
    assert math.isclose(lci.loc[lci.sub_key == "bga", "die_mm2_nom"].iloc[0], 36.0)
