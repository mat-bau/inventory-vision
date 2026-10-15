import json
import math
import sys
from collections import defaultdict

default_class_map = {
    0: "resistors",
    1: "inductors",
    2: "capacitors",
    3: "ICs",
    4: "diodes",
    5: "leds",
    6: "connectors",
    7: "transistors"
}
"""
Default indexes to classes mapping
"""

# ---------------------------------------------------------------------------
# Mass model: each entry is (a_fit, a_pi_low, a_pi_high, b)
#   fit(area)      = a_fit    * area^b
#   PI lower(area) = a_pi_low * area^b
#   PI upper(area) = a_pi_high* area^b
# ---------------------------------------------------------------------------
mass_model_map = {
    #              a_fit        a_low        a_high       b
    "resistors":  (1.378961,   0.947801,    1.944343,    1.205296),
    "inductors":  (3.411289,   1.320510,    7.389738,    1.282065),
    "capacitors": (3.802452,   1.289872,    8.885017,    1.382363),
    "diodes":     (1.495002,   0.626365,    3.080696,    1.046141),
    "leds":       (1.341389,   0.612725,    2.592473,    1.287865),
    "connectors": (0.597460,   0.070947,    2.355680,    1.544120),
    "transistors":(1.950259,   0.800027,    4.066895,    1.275638),
}

keep_classes = set(mass_model_map.keys())


def _mass_nominal(cls, area):
    """Return nominal mass (mg) for *cls* and *area* (mm²)."""
    a, _, _, b = mass_model_map[cls]
    return a * (area ** b)


def _mass_bounds(cls, area_min, area_nom, area_max):
    """
    Return a dict with all uncertainty components for one detection:

        {
            'nom':          nominal mass (mg)                          [a_fit * area_nom^b]

            'model_low':    lower bound from model PI only             [a_low  * area_nom^b]
            'model_high':   upper bound from model PI only             [a_high * area_nom^b]

            'bbox_low':     lower bound from bbox uncertainty only     [a_fit  * area_min^b]
            'bbox_high':    upper bound from bbox uncertainty only     [a_fit  * area_max^b]

            'combined_low': quadratic combination of both lower halves
            'combined_high':quadratic combination of both upper halves
        }

    Quadratic combination (sources assumed independent):
        delta_combined_lo = sqrt(delta_model_lo² + delta_bbox_lo²)
        delta_combined_hi = sqrt(delta_model_hi² + delta_bbox_hi²)

    This is consistent with the quadratic roll-up already used across
    component classes at the board level.
    """
    return mass_bounds_from_coeffs(mass_model_map[cls], area_min, area_nom, area_max)


def mass_bounds_from_coeffs(coeffs, area_min, area_nom, area_max):
    """Same as _mass_bounds, with explicit coefficients (a_fit, a_low, a_high, b).

    Used for subcategory-specific mass models (config/taxonomy.yaml).
    """
    a, a_low, a_high, b = coeffs

    # --- nominal mass ---
    m_nom = a * (area_nom ** b)

    # --- model uncertainty (bbox fixed at nominal) ---
    m_model_low  = a_low  * (area_nom ** b)
    m_model_high = a_high * (area_nom ** b)

    # --- bbox uncertainty (model coefficients fixed at nominal) ---
    m_bbox_low  = a * (area_min ** b)
    m_bbox_high = a * (area_max ** b)

    # --- half-intervals ---
    delta_model_lo = m_nom - m_model_low   # >= 0
    delta_model_hi = m_model_high - m_nom  # >= 0
    delta_bbox_lo  = m_nom - m_bbox_low    # >= 0  (monotone because b > 0)
    delta_bbox_hi  = m_bbox_high - m_nom   # >= 0

    # --- quadratic combination ---
    m_combined_low  = m_nom - math.sqrt(delta_model_lo ** 2 + delta_bbox_lo ** 2)
    m_combined_high = m_nom + math.sqrt(delta_model_hi ** 2 + delta_bbox_hi ** 2)

    return {
        'nom':           m_nom,
        'model_low':     m_model_low,
        'model_high':    m_model_high,
        'bbox_low':      m_bbox_low,
        'bbox_high':     m_bbox_high,
        'combined_low':  max(0.0, m_combined_low),
        'combined_high': m_combined_high,
    }


def normalize_class_name(name):
    """Normalize class name: lowercase and add trailing 's' if absent."""
    n = name.lower()
    if not n.endswith('s'):
        n = n + 's'
    return n


def resolve_class_label(raw_class, class_map):
    try:
        idx = int(raw_class)
    except Exception:
        idx = None
    if idx is not None and idx in class_map:
        return normalize_class_name(class_map[idx])
    if raw_class in class_map:
        return normalize_class_name(class_map[raw_class])
    if idx is not None and str(idx) in class_map:
        return normalize_class_name(class_map[str(idx)])
    return normalize_class_name(raw_class)


def load_class_map(path):
    import json
    try:
        with open(path, "r") as f:
            raw = json.load(f)
            normalized = {}
            for k, v in raw.items():
                try:
                    ik = int(k)
                    normalized[ik] = v
                except Exception:
                    normalized[k] = v
            normalized.update(raw)
            return normalized
    except Exception as e:
        print(f"Error loading class map '{path}': {e}")
        sys.exit(1)


def calculate_mass_per_class(
    component_file,
    class_map=None,
    image_width_mm=1.0,
    image_height_mm=1.0,
    bbox_uncertainty_pct=0.0,
    state_file=None,
):
    """
    Calculate the mass per class for the given component file.

    Parameters
    ----------
    component_file : str
        Path to YOLO-format label file (cropped space).
    class_map : dict, optional
        Mapping {int: str} for class ids.
    image_width_mm : float
        Real-world width of the cropped board (mm).
    image_height_mm : float
        Real-world height of the cropped board (mm).
    bbox_uncertainty_pct : float
        Symmetric relative uncertainty on each bounding-box dimension (%).
        E.g. 5.0 means every width/height is treated as value ± 5%.
        Area scales as dimension², so:
            area_min = area * (1 - pct/100)²
            area_max = area * (1 + pct/100)²

    Returns
    -------
    total_mass       : dict  {cls: nominal_mass_mg}
    counts_per_class : dict  {cls: count}
    total_ic_area    : float
    uncertainty      : dict  {
        'per_class': {
            cls: {
                'model_low':    float,   # mg — model PI only
                'model_high':   float,
                'bbox_low':     float,   # mg — bbox uncertainty only
                'bbox_high':    float,
                'combined_low': float,   # mg — quadratic combination
                'combined_high':float,
                'nom':          float,   # mg — nominal
            }
        },
        'total_nom':          float,   # mg
        'total_model_low':    float,   # mg — quadratic roll-up, model only
        'total_model_high':   float,
        'total_bbox_low':     float,   # mg — quadratic roll-up, bbox only
        'total_bbox_high':    float,
        'total_combined_low': float,   # mg — quadratic roll-up, both sources
        'total_combined_high':float,
    }
    """
    if class_map is None:
        class_map = default_class_map

    # Load modifications from state file if provided
    modifications_per_class = defaultdict(int)
    if state_file:
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            added_by_class = state.get('added_by_class', {})
            deleted_by_class = state.get('deleted_by_class', {})
            reclassified_from_to = state.get('reclassified_from_to', {})
            for cls, count in added_by_class.items():
                modifications_per_class[normalize_class_name(cls)] += count
            for cls, count in deleted_by_class.items():
                modifications_per_class[normalize_class_name(cls)] += count
            for from_cls, to_dict in reclassified_from_to.items():
                for to_cls, count in to_dict.items():
                    modifications_per_class[normalize_class_name(to_cls)] += count
        except Exception as e:
            print(f"Warning: Error loading state file '{state_file}': {e}")
            modifications_per_class = defaultdict(int)
    total_mass       = defaultdict(float)
    total_ic_area    = 0.0
    skipped_counts   = defaultdict(int)
    counts_per_class = defaultdict(int)

    # Per-class accumulators: list of bound dicts per component
    _bounds_list = defaultdict(list)   # cls -> [bound_dict, ...]

    # bbox scale factors
    f_lo       = 1.0 - bbox_uncertainty_pct / 100.0
    f_hi       = 1.0 + bbox_uncertainty_pct / 100.0
    area_f_lo2 = f_lo ** 2
    area_f_hi2 = f_hi ** 2

    with open(component_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                skipped_counts["empty_line"] += 1
                continue
            parts = s.split()
            if len(parts) < 5:
                skipped_counts["malformed"] += 1
                print(f"Warning: line {line_num} malformed, skipping: '{s}'")
                continue

            raw_class, _, _, width_norm, height_norm = parts[:5]
            classes = resolve_class_label(raw_class, class_map)

            try:
                width_mm  = float(width_norm)  * float(image_width_mm)
                height_mm = float(height_norm) * float(image_height_mm)
            except ValueError:
                skipped_counts["invalid_float"] += 1
                print(f"Warning: line {line_num} non-numeric dimensions, skipping.")
                continue

            area_nom = abs(width_mm * height_mm)
            if area_nom <= 0:
                skipped_counts["zero_area"] += 1
                print(f"Warning: line {line_num} zero area, skipping.")
                continue

            area_min = area_nom * area_f_lo2
            area_max = area_nom * area_f_hi2

            # ---- ICs (area-only, no mass model) -------------------------
            if classes == "ics":
                total_ic_area += area_nom
                counts_per_class[classes] += 1
                continue

            # ---- Connectors (general) --------------------
            if classes == "connectors":
                bounds = _mass_bounds("connectors", area_min, area_nom, area_max)
                total_mass["connectors"]       += bounds['nom']
                counts_per_class["connectors"] += 1
                _bounds_list["connectors"].append(bounds)
                continue

            # ---- All other classes --------------------------------------
            if classes not in mass_model_map:
                skipped_counts["no_mass_function"] += 1
                counts_per_class[classes] += 1
                continue

            bounds = _mass_bounds(classes, area_min, area_nom, area_max)

            total_mass[classes]       += bounds['nom']
            counts_per_class[classes] += 1
            _bounds_list[classes].append(bounds)

    # Build per-class uncertainty summary (sum within class, quadratic across)
    per_class_uncertainty = {}

    # Board-level quadratic accumulators for each uncertainty source
    sq_model_lo  = sq_model_hi  = 0.0
    sq_bbox_lo   = sq_bbox_hi   = 0.0
    sq_comb_lo   = sq_comb_hi   = 0.0
    total_nom_all = 0.0

    for cls, bounds_list in _bounds_list.items():
        cls_nom = sum(b['nom'] for b in bounds_list)

        # Combine component-level uncertainties in quadrature rather than linearly.
        # This preserves worst-case aggregation for independent uncertainty sources.
        cls_sq_model_lo = sum((b['nom'] - b['model_low'])   ** 2 for b in bounds_list)
        cls_sq_model_hi = sum((b['model_high'] - b['nom'])  ** 2 for b in bounds_list)
        cls_sq_bbox_lo  = sum((b['nom'] - b['bbox_low'])    ** 2 for b in bounds_list)
        cls_sq_bbox_hi  = sum((b['bbox_high'] - b['nom'])   ** 2 for b in bounds_list)
        cls_sq_comb_lo  = sum((b['nom'] - b['combined_low'])** 2 for b in bounds_list)
        cls_sq_comb_hi  = sum((b['combined_high'] - b['nom'])** 2 for b in bounds_list)

        per_class_uncertainty[cls] = {
            'nom':           cls_nom,
            'model_low':     max(0.0, cls_nom - math.sqrt(cls_sq_model_lo)),
            'model_high':    cls_nom + math.sqrt(cls_sq_model_hi),
            'bbox_low':      max(0.0, cls_nom - math.sqrt(cls_sq_bbox_lo)),
            'bbox_high':     cls_nom + math.sqrt(cls_sq_bbox_hi),
            'combined_low':  max(0.0, cls_nom - math.sqrt(cls_sq_comb_lo)),
            'combined_high': cls_nom + math.sqrt(cls_sq_comb_hi),
        }

        total_nom_all += cls_nom

        # Quadratic accumulators (half-interval per class)
        sq_model_lo  += cls_sq_model_lo
        sq_model_hi  += cls_sq_model_hi
        sq_bbox_lo   += cls_sq_bbox_lo
        sq_bbox_hi   += cls_sq_bbox_hi
        sq_comb_lo   += cls_sq_comb_lo
        sq_comb_hi   += cls_sq_comb_hi

    uncertainty = {
        'per_class':           per_class_uncertainty,
        'total_nom':           total_nom_all,
        'total_model_low':     max(0.0, total_nom_all - math.sqrt(sq_model_lo)),
        'total_model_high':    total_nom_all + math.sqrt(sq_model_hi),
        'total_bbox_low':      max(0.0, total_nom_all - math.sqrt(sq_bbox_lo)),
        'total_bbox_high':     total_nom_all + math.sqrt(sq_bbox_hi),
        'total_combined_low':  max(0.0, total_nom_all - math.sqrt(sq_comb_lo)),
        'total_combined_high': total_nom_all + math.sqrt(sq_comb_hi),
    }

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------
    total_skipped   = sum(skipped_counts.values())
    total_processed = sum(counts_per_class.values()) + total_skipped
    print(f"\nTotal lines inspected: {total_processed}")
    print(f"Total lines skipped:   {total_skipped}")
    for reason, cnt in skipped_counts.items():
        print(f"  {reason}: {cnt}")

    print("\nMass totals with uncertainty breakdown (mg):")
    print(f"  {'class':20s}  {'nom':>10}  "
          f"{'model_lo':>10} {'model_hi':>10}  "
          f"{'bbox_lo':>10}  {'bbox_hi':>10}  "
          f"{'comb_lo':>10}  {'comb_hi':>10}")
    print("  " + "-" * 100)

    for cls in sorted(total_mass.keys()):
        if cls in per_class_uncertainty:
            u = per_class_uncertainty[cls]
            print(
                f"  {cls:20s}  {u['nom']:10.4f}  "
                f"{u['model_low']:10.4f} {u['model_high']:10.4f}  "
                f"{u['bbox_low']:10.4f}  {u['bbox_high']:10.4f}  "
                f"{u['combined_low']:10.4f}  {u['combined_high']:10.4f}"
            )
        else:
            print(f"  {cls:20s}  {total_mass[cls]:10.4f}  (no uncertainty data)")

    print(f"\n  {'TOTAL (quadratic)':20s}  {total_nom_all:10.4f}")
    print(f"  {'  model only':20s}  "
          f"[{uncertainty['total_model_low']:.4f} – {uncertainty['total_model_high']:.4f}]")
    print(f"  {'  bbox only':20s}  "
          f"[{uncertainty['total_bbox_low']:.4f} – {uncertainty['total_bbox_high']:.4f}]")
    print(f"  {'  combined':20s}  "
          f"[{uncertainty['total_combined_low']:.4f} – {uncertainty['total_combined_high']:.4f}]  mg")

    if bbox_uncertainty_pct > 0:
        print(f"  (bbox uncertainty: ±{bbox_uncertainty_pct:.1f}% on each dimension)")

    print("\nComponent counts per class:")
    for cls, c in sorted(counts_per_class.items()):
        print(f"  {cls}: {c}")

    if modifications_per_class:
        print("\nModifications per class:")
        for cls, m in sorted(modifications_per_class.items()):
            print(f"  {cls}: {m}")

    if total_ic_area > 0:
        print(f"\nTotal surface of ICs: {total_ic_area:.3f} mm²")

    return total_mass, counts_per_class, total_ic_area, uncertainty, modifications_per_class