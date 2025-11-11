from __future__ import annotations
from typing import Dict, List, Optional
import json
import re
import urllib.parse
import urllib.request

_PUG_CID_FROM_INCHIKEY = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/cids/JSON"
)
_PUG_VIEW_GHS_BY_CID = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?heading=GHS+Classification"
)

_GHS_CODE_MEANING: Dict[str, str] = {
    "GHS01": "explosive",
    "GHS02": "flammable",
    "GHS03": "oxidizer",
    "GHS04": "gas under pressure",
    "GHS05": "corrosive",
    "GHS06": "toxic",
    "GHS07": "harmful/irritant",
    "GHS08": "health hazard",
    "GHS09": "environmental hazard",
}

def _get_json(url: str, timeout: int = 8) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def cid_from_inchikey(inchikey: str) -> Optional[int]:
    if not inchikey:
        return None
    url = _PUG_CID_FROM_INCHIKEY.format(urllib.parse.quote(inchikey))
    data = _get_json(url)
    cids = (data.get("IdentifierList") or {}).get("CID") or []
    return int(cids[0]) if cids else None

def _dedup(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def ghs_from_cid(cid: int) -> Dict[str, object]:
    try:
        data = _get_json(_PUG_VIEW_GHS_BY_CID.format(int(cid)))
    except Exception:
        return {"hazard_meanings": [], "hazard_label": ""}

    rec = data.get("Record", {})
    meanings: List[str] = []

    def walk(sections):
        for sec in sections or []:
            if sec.get("TOCHeading") == "GHS Classification":
                for inf in sec.get("Information") or []:
                    if inf.get("Name") == "Pictogram(s)":
                        swm = ((inf.get("Value") or {}).get("StringWithMarkup")) or []
                        for entry in swm:
                            for m in entry.get("Markup") or []:
                                if m.get("Type") == "Icon":
                                    url = m.get("URL", "")
                                    m2 = re.search(r"/GHS(\d+)\.svg", url)
                                    if m2:
                                        code = f"GHS{m2.group(1)}"
                                        meaning = _GHS_CODE_MEANING.get(code)
                                        if meaning:
                                            meanings.append(meaning)
            walk(sec.get("Section"))

    walk(rec.get("Section"))
    meanings = _dedup(meanings)
    return {
        "hazard_meanings": meanings,
        "hazard_label": ", ".join(meanings) if meanings else "",
    }

def ghs_from_inchikey(inchikey: str) -> Dict[str, object]:
    cid = cid_from_inchikey(inchikey)
    if cid is None:
        return {"hazard_meanings": [], "hazard_label": ""}
    return ghs_from_cid(cid)
