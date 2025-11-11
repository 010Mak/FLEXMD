from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re
import requests

_DEFAULT_TIMEOUT = 4.0
_MAX_SYNONYMS_RETURNED = 25

def _make_session(user_agent: str = "mmd-sim/0.1 (naming)") -> requests.Session:
    try:
        import requests_cache
        sess = requests_cache.CachedSession(
            cache_name="chem-names-cache", backend="sqlite", expire_after=24 * 3600
        )
    except Exception:
        sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    return sess

_session = _make_session()

_WS = re.compile(r"\s+")
def _clean(s: str) -> str:
    s = s.strip()
    s = _WS.sub(" ", s)
    return s

def _dedupe_keep_order(strings: List[str]) -> List[str]:
    seen, out = set(), []
    for s in strings:
        t = s.strip()
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out

def _is_probably_iupac(name: str) -> bool:
    s = name.lower()
    if any(tok in s for tok in ["(", ")", "[", "]", "alpha", "beta", "omega", "cis", "trans"]):
        return True
    if any(c.isdigit() for c in s):
        return True
    if "-yl" in s or "oxo" in s or "hydroxy" in s or "amino" in s:
        return True
    return False

def _score_commonness(name: str) -> float:
    s = name.strip()
    if not s:
        return -1e9
    L = len(s)
    score = 0.0
    score += max(0, 40 - L) / 40.0
    for suf in (" acid", " alcohol", "ol", "one", "ane", "ene", "ose", "amide"):
        if s.lower().endswith(suf):
            score += 0.15
            break
    if _is_probably_iupac(s):
        score -= 0.8
    if s.isupper() and L > 3:
        score -= 0.5
    return score

def _pick_common_name(title: Optional[str], iupac: Optional[str], synonyms: List[str]) -> Tuple[Optional[str], str]:
    if title:
        return _clean(title), "pubchem"

    iupac_lc = (iupac or "").lower()
    cands = [s for s in synonyms if s.strip()]
    cands = [s for s in cands if s.lower() != iupac_lc]
    cands = _dedupe_keep_order(cands)

    if not cands and iupac:
        return _clean(iupac), "pubchem"

    if cands:
        best = max(cands, key=_score_commonness)
        return _clean(best), ""

    return None, ""


def _pubchem_properties_by_inchikey(inchikey: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/Title,IUPACName/JSON"
    r = _session.get(url, timeout=timeout)
    if r.status_code != 200:
        return None, None
    data = r.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None, None
    rec = props[0]
    title = rec.get("Title")
    iupac = rec.get("IUPACName") or rec.get("IUPACNameTrade") or rec.get("IUPACNamePreferred")
    return title, iupac

def _pubchem_synonyms_by_inchikey(inchikey: str, timeout: float) -> List[str]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/synonyms/JSON"
    r = _session.get(url, timeout=timeout)
    if r.status_code != 200:
        return []
    data = r.json()
    info = data.get("InformationList", {}).get("Information", [])
    if not info:
        return []
    syns = info[0].get("Synonym", []) or []
    return _dedupe_keep_order([_clean(s) for s in syns])[:_MAX_SYNONYMS_RETURNED]


def _nci_text(endpoint: str, timeout: float) -> Optional[str]:
    r = _session.get(endpoint, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.text.strip()

def _nci_iupac(identifier: str, timeout: float) -> Optional[str]:
    url = f"https://cactus.nci.nih.gov/chemical/structure/{identifier}/iupac_name"
    txt = _nci_text(url, timeout)
    if not txt or txt.lower().startswith("404") or "not found" in txt.lower():
        return None
    return _clean(txt)

def _nci_synonyms(identifier: str, timeout: float) -> List[str]:
    url = f"https://cactus.nci.nih.gov/chemical/structure/{identifier}/names"
    txt = _nci_text(url, timeout)
    if not txt:
        return []
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return _dedupe_keep_order([_clean(s) for s in lines])[:_MAX_SYNONYMS_RETURNED]


def resolve_element_name(symbol: str, timeout: float = _DEFAULT_TIMEOUT) -> Dict[str, Optional[str]]:
    sym = symbol.capitalize()
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON"
        r = _session.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            for row in data.get("Table", {}).get("Row", []):
                cells = row.get("Cell", [])
                if len(cells) >= 3 and str(cells[1]).strip().capitalize() == sym:
                    return {"common_name": _clean(str(cells[2])), "name_source": "pubchem"}
    except Exception:
        pass

    try:
        from mendeleev import element as _me
        name = _me(sym).name
        if name:
            return {"common_name": _clean(name), "name_source": "mendeleev"}
    except Exception:
        pass

    return {"common_name": None, "name_source": ""}


def resolve_molecule_names(
    inchikey: Optional[str] = None,
    smiles: Optional[str] = None,
    inchi: Optional[str] = None,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    include_synonyms: bool = True,
) -> Dict[str, Optional[str] | List[str]]:

    title: Optional[str] = None
    iupac: Optional[str] = None
    syns: List[str] = []

    ik = (inchikey or "").strip()
    if ik:
        try:
            title, iupac = _pubchem_properties_by_inchikey(ik, timeout)
            if include_synonyms and not title:
                syns = _pubchem_synonyms_by_inchikey(ik, timeout)
        except Exception:
            pass

    cir_id = ik or (smiles or "") or (inchi or "")
    cir_id = cir_id.strip()
    used_cir = False
    if cir_id and (title is None and iupac is None) or (include_synonyms and not syns):
        try:
            if iupac is None:
                iupac = _nci_iupac(cir_id, timeout)
            if include_synonyms and not syns:
                syns = _nci_synonyms(cir_id, timeout)
            used_cir = True
        except Exception:
            pass

    common, src = _pick_common_name(title, iupac, syns)
    if used_cir and src == "":
        src = "nci"
    elif common and src == "" and title:
        src = "pubchem"

    result: Dict[str, Optional[str] | List[str]] = {
        "common_name": common,
        "iupac_name": _clean(iupac) if iupac else None,
        "name_source": src,
    }
    if include_synonyms:
        result["synonyms"] = syns[:_MAX_SYNONYMS_RETURNED]
    return result
