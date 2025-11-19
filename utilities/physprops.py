from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from utilities.ghs import cid_from_inchikey


_PUG_VIEW_General_BY_CID = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
)

_PUG_VIEW_BY_HEADING = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?heading={}"
)


def _get_json(url: str, timeout: int = 8) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CFKcfk])?")


def _text_to_temp_K(text: str) -> Optional[float]:
    if not text:
        return None

    match = _TEMP_RE.search(text)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).upper() if match.group(2) else None

    if "K" in text.upper() or unit == "K":
        return value

    if "°F" in text.upper() or unit == "F":
        return (value - 32.0) * 5.0 / 9.0 + 273.15

    return value + 273.15


def _extract_temps_from_record(data: dict) -> List[float]:
    rec = data.get("Record") or {}
    temps: List[float] = []

    def walk_sections(sections: Any) -> None:
        for sec in sections or []:
            for inf in sec.get("Information") or []:
                val = inf.get("Value") or {}
                swm = val.get("StringWithMarkup") or []
                for entry in swm:
                    s = entry.get("String")
                    if not isinstance(s, str):
                        continue
                    tK = _text_to_temp_K(s)
                    if tK is not None:
                        temps.append(tK)
            walk_sections(sec.get("Section"))

    walk_sections(rec.get("Section"))
    return temps


def _fetch_heading_temps(cid: int, heading: str) -> List[float]:
    try:
        url = _PUG_VIEW_BY_HEADING.format(int(cid), urllib.parse.quote(heading))
        data = _get_json(url)
    except Exception:
        return []
    return _extract_temps_from_record(data)


def _best_temp_for_property(cid: int, heading_candidates: List[str]) -> Optional[float]:
    for heading in heading_candidates:
        temps = _fetch_heading_temps(cid, heading)
        if temps:
            return float(temps[0])
    return None


def physprops_from_inchikey(inchikey: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if not inchikey:
        return props

    cid = cid_from_inchikey(inchikey)
    if cid is None:
        return props

    mpK = _best_temp_for_property(cid, ["Melting Point"])
    bpK = _best_temp_for_property(cid, ["Boiling Point"])

    if mpK is not None:
        props["melting_point"] = {
            "value_K": mpK,
            "value_C": mpK - 273.15,
            "source": "PubChem PUG View",
        }
        props["freezing_point"] = {
            "value_K": mpK,
            "value_C": mpK - 273.15,
            "alias_of": "melting_point",
            "source": "PubChem PUG View",
        }

    if bpK is not None:
        props["boiling_point"] = {
            "value_K": bpK,
            "value_C": bpK - 273.15,
            "source": "PubChem PUG View",
        }

    return props


def build_phase_curve_1atm(
    *,
    melting_point_K: float,
    boiling_point_K: float,
    t_min_K: Optional[float] = None,
    t_max_K: Optional[float] = None,
    n_points: int = 200,
) -> Dict[str, Any]:
    mpK = float(melting_point_K)
    bpK = float(boiling_point_K)
    if n_points < 3:
        n_points = 3

    if t_min_K is None:
        t_min_K = max(1.0, mpK - 100.0)
    if t_max_K is None:
        t_max_K = bpK + 100.0

    t_min = float(t_min_K)
    t_max = float(t_max_K)
    if t_max <= t_min:
        t_max = t_min + 10.0

    points: List[Dict[str, Any]] = []
    span = t_max - t_min
    step = span / (n_points - 1)

    for i in range(n_points):
        T = t_min + step * i
        if T < mpK:
            phase = "solid"
            phase_code = 0.0
        elif T < bpK:
            phase = "liquid"
            phase_code = 1.0
        else:
            phase = "gas"
            phase_code = 2.0
        points.append(
            {
                "T_K": T,
                "phase": phase,
                "phase_code": phase_code,
            }
        )

    points.append(
        {
            "T_K": mpK,
            "phase": "solid_liquid_boundary",
            "phase_code": 0.5,
        }
    )
    points.append(
        {
            "T_K": bpK,
            "phase": "liquid_gas_boundary",
            "phase_code": 1.5,
        }
    )

    points.sort(key=lambda p: p["T_K"])

    return {
        "kind": "T_vs_phase_1atm",
        "pressure_Pa": 101325.0,
        "t_min_K": t_min,
        "t_max_K": t_max,
        "n_points": len(points),
        "legend": {
            "0.0": "solid",
            "0.5": "solid_liquid_boundary",
            "1.0": "liquid",
            "1.5": "liquid_gas_boundary",
            "2.0": "gas",
        },
        "points": points,
    }
