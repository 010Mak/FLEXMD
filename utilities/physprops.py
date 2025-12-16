from __future__ import annotations

import json
import math
import re
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional

from utilities.ghs import cid_from_inchikey


_PUG_VIEW_BY_CID = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
_PUG_VIEW_BY_HEADING = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={heading}"
)


def _get_json(url: str, timeout: int = 8) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))



_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CFKcfk])?")


def _text_to_temp_K(text: str) -> Optional[float]:
    if not text:
        return None
    m = _TEMP_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else None
    upper = text.upper()
    if "K" in upper or unit == "K":
        return value
    if "F" in upper or unit == "F":
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
        url = _PUG_VIEW_BY_HEADING.format(
            cid=int(cid),
            heading=urllib.parse.quote(heading),
        )
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



def _extract_strings_from_record(data: dict) -> List[str]:
    rec = data.get("Record") or {}
    out: List[str] = []

    def walk(sections: Any) -> None:
        for sec in sections or []:
            for inf in sec.get("Information") or []:
                val = inf.get("Value") or {}
                swm = val.get("StringWithMarkup") or []
                for entry in swm:
                    s = entry.get("String")
                    if isinstance(s, str):
                        out.append(s)
            walk(sec.get("Section"))

    walk(rec.get("Section"))
    return out


def _fetch_heading_strings(cid: int, heading: str) -> List[str]:
    try:
        url = _PUG_VIEW_BY_HEADING.format(
            cid=int(cid),
            heading=urllib.parse.quote(heading),
        )
        data = _get_json(url)
    except Exception:
        return []
    return _extract_strings_from_record(data)



def _extract_description(cid: int) -> Optional[str]:
    headings = [
        "Description",
        "Chemical Safety",
        "Drug and Medication Information",
    ]
    for heading in headings:
        strs = _fetch_heading_strings(cid, heading)
        if strs:
            for s in strs:
                s = s.strip()
                if len(s) > 20:
                    return s
    return None



def _extract_nfpa_from_strings(texts: List[str]) -> Dict[str, Any]:
    health: Optional[int] = None
    flamm: Optional[int] = None
    instab: Optional[int] = None
    specials: List[str] = []

    for s in texts or []:
        lower = s.lower()

        m = re.search(r"(health\s*(hazard)?\s*(rating)?)[^0-9]*([0-4])", lower)
        if m and health is None:
            try:
                health = int(m.group(4))
            except Exception:
                pass

        m = re.search(r"(flammability|fire\s*(hazard)?\s*(rating)?)[^0-9]*([0-4])", lower)
        if m and flamm is None:
            try:
                flamm = int(m.group(4))
            except Exception:
                pass

        m = re.search(r"(reactivity|instability\s*(rating)?)[^0-9]*([0-4])", lower)
        if m and instab is None:
            try:
                instab = int(m.group(3))
            except Exception:
                pass

        if "special" in lower:
            tokens = re.findall(r"\b[A-Z]{1,3}\b", s)
            for tok in tokens:
                if tok not in specials:
                    specials.append(tok)

    nfpa: Dict[str, Any] = {
        "health": health,
        "flammability": flamm,
        "instability": instab,
        "special": specials,
    }
    if health is None and flamm is None and instab is None and not specials:
        return {}
    return nfpa


def _fetch_nfpa(cid: int) -> Dict[str, Any]:
    try:
        url = _PUG_VIEW_BY_CID.format(cid=int(cid))
        data = _get_json(url)
    except Exception:
        return {}

    rec = data.get("Record") or {}
    texts: List[str] = []

    health: Optional[int] = None
    flamm: Optional[int] = None
    instab: Optional[int] = None
    specials: List[str] = []
    code_digits: Optional[List[int]] = None

    def _first_digit_0_to_4(s: str) -> Optional[int]:
        m = re.search(r"\b([0-4])\b", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def walk(sections: Any) -> None:
        nonlocal health, flamm, instab, specials, code_digits, texts
        for sec in sections or []:
            heading = (sec.get("TOCHeading") or "")
            if "nfpa" in heading.lower():
                for inf in sec.get("Information") or []:
                    name = (inf.get("Name") or "").lower()
                    val = inf.get("Value") or {}
                    swm = val.get("StringWithMarkup") or []

                    for entry in swm:
                        s = entry.get("String")
                        if isinstance(s, str):
                            texts.append(s)

                    for entry in swm:
                        for mk in entry.get("Markup") or []:
                            extra = (mk.get("Extra") or "").strip()
                            if not extra:
                                continue
                            parts = re.split(r"\s*-\s*", extra)
                            digits: List[int] = []
                            extra_specials: List[str] = []
                            for part in parts:
                                part = part.strip()
                                if re.fullmatch(r"[0-4]", part):
                                    digits.append(int(part))
                                elif re.fullmatch(r"[A-Za-z]{1,3}", part):
                                    extra_specials.append(part)
                            if len(digits) >= 3:
                                code_digits = digits[:3]
                            for c in extra_specials:
                                if c not in specials:
                                    specials.append(c)

                    full_text = " ".join(
                        entry.get("String") or "" for entry in swm
                        if isinstance(entry.get("String"), str)
                    )
                    lower_text = full_text.lower()

                    if "health" in name:
                        if health is None:
                            v = _first_digit_0_to_4(lower_text)
                            if v is not None:
                                health = v
                    elif "fire" in name or "flamm" in name:
                        if flamm is None:
                            v = _first_digit_0_to_4(lower_text)
                            if v is not None:
                                flamm = v
                    elif "instability" in name or "reactivity" in name:
                        if instab is None:
                            v = _first_digit_0_to_4(lower_text)
                            if v is not None:
                                instab = v
                    elif "special" in name or "specific" in name:
                        codes = re.findall(r"\b[A-Z]{1,3}\b", full_text)
                        for c in codes:
                            if c not in specials:
                                specials.append(c)

            walk(sec.get("Section"))

    walk(rec.get("Section"))

    if code_digits and len(code_digits) == 3:
        if health is None:
            health = code_digits[0]
        if flamm is None:
            flamm = code_digits[1]
        if instab is None:
            instab = code_digits[2]

    if health is None and flamm is None and instab is None and not specials and texts:
        fallback = _extract_nfpa_from_strings(texts)
        if fallback:
            fallback["source"] = "PubChem PUG-View (NFPA textual fallback)"
            return fallback

    if health is None and flamm is None and instab is None and not specials:
        return {}

    return {
        "health": health,
        "flammability": flamm,
        "instability": instab,
        "special": specials,
        "source": "PubChem PUG-View (NFPA section)",
    }



_NUM_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*([A-Za-zµμ/%·\^0-9\-]+))?")


def _parse_value_and_temp(text: str) -> Dict[str, Any]:

    if not text:
        return {"value": None, "unit": None, "temp_K": None, "temp_C": None}

    val = None
    unit = None
    m_val = _NUM_UNIT_RE.search(text)
    if m_val:
        try:
            val = float(m_val.group(1))
        except Exception:
            val = None
        unit = (m_val.group(2) or "").strip() or None

    tK = None
    tC = None
    m_temp = _TEMP_RE.search(text)
    if m_temp:
        try:
            t_value = float(m_temp.group(1))
            u = m_temp.group(2).upper() if m_temp.group(2) else None
            if u == "K":
                tK = t_value
                tC = t_value - 273.15
            elif u == "F":
                tC = (t_value - 32.0) * 5.0 / 9.0
                tK = tC + 273.15
            else:
                tC = t_value
                tK = t_value + 273.15
        except Exception:
            tK = None
            tC = None

    return {"value": val, "unit": unit, "temp_K": tK, "temp_C": tC}


def _parse_temp_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _TEMP_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except Exception:
        return None
    unit = m.group(2).upper() if m.group(2) else None
    u = unit or ""
    if "K" in u:
        valK = value
        valC = value - 273.15
        unit_out = "K"
    elif "F" in u:
        valC = (value - 32.0) * 5.0 / 9.0
        valK = valC + 273.15
        unit_out = "°F"
    else:
        valC = value
        valK = value + 273.15
        unit_out = "°C"

    return {
        "value_K": valK,
        "value_C": valC,
        "unit": unit_out,
        "raw": text,
    }


def _fetch_single_property_text(cid: int, heading_candidates: List[str]) -> Optional[str]:
    for heading in heading_candidates:
        texts = _fetch_heading_strings(cid, heading)
        if texts:
            joined = " ".join(t.strip() for t in texts if t.strip())
            if joined:
                return joined
    return None


def _extract_physical_props(cid: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    density_text = _fetch_single_property_text(cid, ["Density"])
    if density_text:
        vp = _parse_value_and_temp(density_text)
        out["density"] = {
            "value": vp["value"],
            "unit": vp["unit"],
            "temp_K": vp["temp_K"],
            "temp_C": vp["temp_C"],
            "raw": density_text,
            "source": "PubChem Experimental Properties",
        }

    flash_text = _fetch_single_property_text(cid, ["Flash Point"])
    if flash_text:
        block = _parse_temp_block(flash_text)
        if block:
            block["source"] = "PubChem Experimental Properties"
            out["flash_point"] = block
        else:
            out["flash_point"] = {
                "raw": flash_text,
                "source": "PubChem Experimental Properties",
            }

    auto_text = _fetch_single_property_text(
        cid, ["Autoignition Temperature", "Auto Ignition Temperature"]
    )
    if auto_text:
        block = _parse_temp_block(auto_text)
        if block:
            block["source"] = "PubChem Experimental Properties"
            out["autoignition_temperature"] = block
        else:
            out["autoignition_temperature"] = {
                "raw": auto_text,
                "source": "PubChem Experimental Properties",
            }

    vap_text = _fetch_single_property_text(cid, ["Vapor Pressure", "Vapour Pressure"])
    if vap_text:
        vp = _parse_value_and_temp(vap_text)
        out["vapor_pressure"] = {
            "value": vp["value"],
            "unit": vp["unit"],
            "temp_K": vp["temp_K"],
            "temp_C": vp["temp_C"],
            "raw": vap_text,
            "source": "PubChem Experimental Properties",
        }

    visc_text = _fetch_single_property_text(cid, ["Viscosity"])
    if visc_text:
        vp = _parse_value_and_temp(visc_text)
        out["viscosity"] = {
            "value": vp["value"],
            "unit": vp["unit"],
            "temp_K": vp["temp_K"],
            "temp_C": vp["temp_C"],
            "raw": visc_text,
            "source": "PubChem Experimental Properties",
        }

    refr_text = _fetch_single_property_text(cid, ["Refractive Index"])
    if refr_text:
        m = re.search(r"(-?\d+(?:\.\d+)?)", refr_text)
        val = None
        if m:
            try:
                val = float(m.group(1))
            except Exception:
                val = None
        out["refractive_index"] = {
            "value": val,
            "raw": refr_text,
            "source": "PubChem Experimental Properties",
        }

    return out



def physprops_from_inchikey(inchikey: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if not inchikey:
        return props

    cid = cid_from_inchikey(inchikey)
    if cid is None:
        return props

    mpK = _best_temp_for_property(
        cid,
        ["Melting Point", "Melting point", "Fusion", "Freezing Point"],
    )
    bpK = _best_temp_for_property(
        cid,
        ["Boiling Point", "Boiling point", "Boiling Pt"],
    )

    if mpK is not None:
        props["melting_point"] = {
            "value_K": mpK,
            "value_C": mpK - 273.15,
            "source": "PubChem PUG-View",
        }
        props["freezing_point"] = {
            "value_K": mpK,
            "value_C": mpK - 273.15,
            "alias_of": "melting_point",
            "source": "PubChem PUG-View",
        }

    if bpK is not None:
        props["boiling_point"] = {
            "value_K": bpK,
            "value_C": bpK - 273.15,
            "source": "PubChem PUG-View",
        }

    desc = _extract_description(cid)
    if desc:
        props["description"] = {
            "text": desc,
            "source": "PubChem PUG-View Description / Safety",
        }

    nfpa = _fetch_nfpa(cid)
    if nfpa:
        props["nfpa704"] = nfpa

    phys = _extract_physical_props(cid)
    for k, v in (phys or {}).items():
        props[k] = v

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
