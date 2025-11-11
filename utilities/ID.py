from __future__ import annotations

import json
import math
import hashlib
from typing import Dict, List, Tuple, Optional, Any

try:
    from rdkit import Chem
    from rdkit.Chem import inchi
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False



def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _canon_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _canon_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canon_dict(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canon_dict(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj



def _rdkit_mol_from_graph(elements: List[str],
                          bonds0: List[Tuple[int, int]],
                          charges: Optional[List[int]] = None) -> Chem.Mol:
    mol = Chem.RWMol()
    for i, el in enumerate(elements):
        a = Chem.Atom(el)
        if charges is not None:
            try:
                a.SetFormalCharge(int(charges[i] or 0))
            except Exception:
                a.SetFormalCharge(0)
        mol.AddAtom(a)
    for (i, j) in bonds0:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        mol.AddBond(int(a), int(b), Chem.rdchem.BondType.SINGLE)
    m = mol.GetMol()
    Chem.SanitizeMol(m, catchErrors=True)
    return m

def _graph_id_rdkit(elements: List[str],
                    bonds0: List[Tuple[int, int]],
                    charges: Optional[List[int]] = None) -> Tuple[str, List[int]]:
    m = _rdkit_mol_from_graph(elements, bonds0, charges)
    ranks = list(Chem.CanonicalRankAtoms(m))
    order = list(range(len(elements)))
    order.sort(key=lambda i: (ranks[i], i))
    try:
        ik = inchi.MolToInchiKey(m)
        if ik and isinstance(ik, str):
            gid = "inchikey:" + ik
        else:
            raise RuntimeError("No InChIKey")
    except Exception:
        smi = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
        gid = "sha256:" + _sha256_hex(smi.encode("utf-8"))
    return gid, order

def _graph_id_fallback(elements: List[str],
                       bonds0: List[Tuple[int, int]]) -> Tuple[str, List[int]]:
    n = len(elements)
    E = sorted(set((min(i, j), max(i, j)) for i, j in bonds0 if i != j))
    labels = [str(elements[i]).capitalize() for i in range(n)]
    nbrs = [[] for _ in range(n)]
    for a, b in E:
        nbrs[a].append(b)
        nbrs[b].append(a)
    for i in range(n):
        nbrs[i].sort()
    last = None
    for _ in range(min(12, max(3, int(math.log2(max(1, n))) + 3))):
        sigs = []
        for i in range(n):
            ns = ",".join(labels[j] for j in nbrs[i])
            sigs.append(_sha256_hex(f"{labels[i]}|[{ns}]".encode()))
        if sigs == last:
            break
        last = labels = sigs
    order = list(range(n))
    order.sort(key=lambda i: (labels[i], i))
    payload = {
        "v": "mmdfled.wl.v1",
        "N": [elements[i].capitalize() for i in order],
        "E": [f"{min(a,b)}-{max(a,b)}" for a, b in E],
    }
    gid = "sha256:" + _sha256_hex(_canon_json(payload))
    return gid, order


def canonical_graph_id(elements: List[str],
                       bonds0: List[Tuple[int, int]],
                       charges: Optional[List[int]] = None) -> Tuple[str, List[int]]:
    if _HAS_RDKIT:
        try:
            return _graph_id_rdkit(elements, bonds0, charges)
        except Exception:
            pass
    return _graph_id_fallback(elements, bonds0)



def conformer_id(elements: List[str],
                 positionsA: List[Tuple[float, float, float]],
                 canon_order: Optional[List[int]] = None,
                 q: float = 0.01) -> str:
    n = len(elements)
    if n == 0:
        return "sha256:" + _sha256_hex(b"empty")
    if canon_order is None:
        canon_order = list(range(n))
    els = [elements[i].capitalize() for i in canon_order]
    pos = [positionsA[i] for i in canon_order]

    cx = sum(p[0] for p in pos) / n
    cy = sum(p[1] for p in pos) / n
    cz = sum(p[2] for p in pos) / n
    pos0 = [(p[0]-cx, p[1]-cy, p[2]-cz) for p in pos]

    dists = []
    for i in range(n - 1):
        xi, yi, zi = pos0[i]
        for j in range(i + 1, n):
            xj, yj, zj = pos0[j]
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            dq = round(d / q) * q
            dists.append(dq)
    dists.sort()

    payload = {"v": "mmdfled.conf.v1", "els": els, "dists": dists}
    return "sha256:" + _sha256_hex(_canon_json(payload))

def _components_from_bonds(n_atoms: int, bonds0: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n_atoms)]
    for i, j in bonds0:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        adj[a].append(b); adj[b].append(a)
    for k in range(n_atoms):
        adj[k].sort()
    seen = [False] * n_atoms
    comps: List[List[int]] = []
    for i in range(n_atoms):
        if not seen[i]:
            stack = [i]; seen[i] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adj[v]:
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
            comp.sort()
            comps.append(comp)
    return comps

def build_method_signature(*,
                           backend: str,
                           thermostat_name: Optional[str],
                           dt_ps: float,
                           n_steps: int,
                           report_stride: int,
                           plugin_args: Optional[Dict[str, Any]] = None,
                           ff_name: Optional[str] = None,
                           ff_param_sha256: Optional[str] = None,
                           feature_flags: Optional[List[str]] = None) -> Dict[str, Any]:
    ms = {
        "backend": str(backend or "").lower(),
        "thermostat": (str(thermostat_name).lower() if thermostat_name else None),
        "timestep_ps": round(float(dt_ps), 12),
        "n_steps": int(n_steps),
        "report_stride": int(report_stride),
        "ff": {
            "name": ff_name or None,
            "param_sha256": ff_param_sha256 or None,
        },
        "plugin_args": _canon_dict(plugin_args or {}),
        "feature_flags": sorted(feature_flags or []),
    }
    return ms

def result_id(graph_id: str,
              method_signature: Dict[str, Any],
              scope: str = "graph",
              conditions: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "v": "mmdfled.res.v1",
        "scope": str(scope or "graph"),
        "graph_id": graph_id,
        "method": _canon_dict(method_signature),
        "conditions": _canon_dict(conditions or {}),
    }
    return "sha256:" + _sha256_hex(_canon_json(payload))

def compute_all_ids(*,
                    elements: List[str],
                    positionsA: List[Tuple[float, float, float]],
                    bonds0: List[Tuple[int, int]],
                    backend: str,
                    thermostat_name: Optional[str],
                    dt_ps: float,
                    n_steps: int,
                    report_stride: int,
                    plugin_args: Optional[Dict[str, Any]] = None,
                    scope: str = "graph",
                    default_tempK: Optional[float] = None,
                    charges: Optional[List[int]] = None,
                    ff_name: Optional[str] = None,
                    ff_param_sha256: Optional[str] = None,
                    feature_flags: Optional[List[str]] = None) -> Dict[str, Any]:
    gid, order = canonical_graph_id(elements, bonds0, charges=charges)
    cid = conformer_id(elements, positionsA, canon_order=order, q=0.01)

    n = len(elements)
    comps = _components_from_bonds(n, bonds0)
    frags: List[Dict[str, Any]] = []
    for comp in comps:
        remap = {old: i for i, old in enumerate(comp)}
        f_elems = [elements[i] for i in comp]
        f_posA  = [positionsA[i] for i in comp]
        f_bonds = []
        for (i, j) in bonds0:
            if i in remap and j in remap:
                a, b = remap[i], remap[j]
                if a > b:
                    a, b = b, a
                f_bonds.append((a, b))
        f_bonds = sorted(set(f_bonds))
        f_gid, f_order = canonical_graph_id(f_elems, f_bonds)
        f_cid = conformer_id(f_elems, f_posA, canon_order=f_order, q=0.01)
        frags.append({
            "indices": comp,
            "graph_id": f_gid,
            "conformer_id": f_cid,
        })

    mixture_id = None
    if len(frags) > 1:
        bag = sorted(f["graph_id"] for f in frags)
        mixture_id = "sha256:" + _sha256_hex(_canon_json({"mix": bag}))

    ms = build_method_signature(
        backend=backend,
        thermostat_name=thermostat_name,
        dt_ps=dt_ps,
        n_steps=n_steps,
        report_stride=report_stride,
        plugin_args=plugin_args,
        ff_name=ff_name,
        ff_param_sha256=ff_param_sha256,
        feature_flags=feature_flags,
    )
    cond = {"temperature_K": float(default_tempK)} if default_tempK is not None else {}
    rid = result_id(gid, ms, scope=scope, conditions=cond)

    return {
        "graph_id": gid,
        "conformer_id": cid,
        "fragments": frags,
        "fragment_count": len(frags),
        "mixture_id": mixture_id,
        "method_signature": ms,
        "result_id": rid,
    }
