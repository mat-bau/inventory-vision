"""
Component taxonomy: macro classes predicted by the detector and the
subcategories a human can assign after inference.

The taxonomy is read from ``config/taxonomy.yaml``. Every macro class gets an
implicit ``unspecified`` subcategory first, which inherits all macro settings
and therefore reproduces the original InventoryVision behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

UNSPECIFIED = "unspecified"

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "config" / "taxonomy.yaml"


def normalize_name(name) -> str:
    """Same normalisation as component_to_mass.normalize_class_name."""
    n = str(name).strip().lower()
    if not n.endswith("s"):
        n += "s"
    return n


@dataclass(frozen=True)
class DPR:
    nominal: float
    low: float
    high: float
    source: str = ""
    range_reported: bool = True

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        nom = float(d["nominal"])
        has_range = d.get("low") is not None and d.get("high") is not None
        low = float(d["low"]) if has_range else nom
        high = float(d["high"]) if has_range else nom
        if not (0 < low <= nom <= high <= 1):
            raise ValueError(f"Invalid DPR (need 0 < low <= nominal <= high <= 1): {d}")
        return cls(nom, low, high, str(d.get("source", "")), has_range)


@dataclass(frozen=True)
class Subcategory:
    key: str
    label: str
    short: str
    ecoinvent_flow: str | None
    flow_verified: bool
    mass_model: tuple | None = None
    dpr: DPR | None = None
    stack_count: int = 1
    note: str = ""

    @property
    def is_unspecified(self) -> bool:
        return self.key == UNSPECIFIED


@dataclass
class MacroClass:
    key: str
    label: str
    model_names: list
    subcategories: dict = field(default_factory=dict)  # key -> Subcategory (ordered)
    note: str = ""

    def sub(self, key) -> Subcategory:
        return self.subcategories.get(key or UNSPECIFIED, self.subcategories[UNSPECIFIED])

    def sub_list(self) -> list:
        return list(self.subcategories.values())


def _mass_model(value, where):
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{where}: mass_model needs 4 values [a_fit, a_low, a_high, b]")
    return tuple(float(v) for v in value)


class Taxonomy:
    def __init__(self, classes: dict, path: Path | None = None):
        self.classes = classes  # key -> MacroClass
        self.path = path
        self._by_name = {}
        for mc in classes.values():
            for n in [mc.key, mc.label, *mc.model_names]:
                self._by_name[normalize_name(n)] = mc

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path=None) -> Taxonomy:
        path = Path(path) if path else DEFAULT_TAXONOMY_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, path)

    @classmethod
    def from_dict(cls, raw: dict, path=None) -> Taxonomy:
        classes = {}
        for key, c in (raw.get("classes") or {}).items():
            macro_flow = c.get("ecoinvent_flow")
            macro_verified = bool(c.get("flow_verified", False))
            macro_dpr = DPR.from_dict(c.get("dpr"))
            macro_mass = _mass_model(c.get("mass_model"), key)
            macro_stack = int(c.get("stack_count", 1))
            label = c.get("label", key)

            subs = {
                UNSPECIFIED: Subcategory(
                    key=UNSPECIFIED,
                    label="Unspecified",
                    short="",
                    ecoinvent_flow=macro_flow,
                    flow_verified=macro_verified,
                    mass_model=macro_mass,
                    dpr=macro_dpr,
                    stack_count=macro_stack,
                    note=c.get("note", ""),
                )
            }
            for skey, s in (c.get("subcategories") or {}).items():
                if skey == UNSPECIFIED:
                    raise ValueError(f"'{UNSPECIFIED}' is reserved (class {key})")
                s = s or {}
                has_own_flow = "ecoinvent_flow" in s
                subs[skey] = Subcategory(
                    key=skey,
                    label=s.get("label", skey),
                    short=s.get("short", skey),
                    ecoinvent_flow=s.get("ecoinvent_flow") if has_own_flow else macro_flow,
                    flow_verified=bool(s.get("flow_verified", False)) if has_own_flow else macro_verified,
                    mass_model=_mass_model(s.get("mass_model"), f"{key}.{skey}") or macro_mass,
                    dpr=DPR.from_dict(s.get("dpr")) or macro_dpr,
                    stack_count=int(s.get("stack_count", macro_stack)),
                    note=s.get("note", ""),
                )
            classes[key] = MacroClass(
                key=key,
                label=label,
                model_names=list(c.get("model_names") or []),
                subcategories=subs,
                note=c.get("note", ""),
            )
        return cls(classes, Path(path) if path else None)

    # ---------------------------------------------------------------- lookup
    def macro(self, model_class_name) -> MacroClass | None:
        return self._by_name.get(normalize_name(model_class_name))

    def subcategories(self, model_class_name) -> list:
        mc = self.macro(model_class_name)
        return mc.sub_list() if mc else []

    def sub(self, model_class_name, sub_key) -> Subcategory | None:
        mc = self.macro(model_class_name)
        return mc.sub(sub_key) if mc else None

    def is_valid_sub(self, model_class_name, sub_key) -> bool:
        mc = self.macro(model_class_name)
        return bool(mc) and (sub_key or UNSPECIFIED) in mc.subcategories


def load_taxonomy(path=None) -> Taxonomy:
    return Taxonomy.load(path)
