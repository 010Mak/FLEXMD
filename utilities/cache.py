
from __future__ import annotations

import os
import io
import json
import time
import math
import threading
import hashlib
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

try:
    from utilities.config import (
        CACHE_MAX_MB,
        CACHE_TTL_DAYS,
        CACHE_ENABLE,
        CACHE_MODE,
    )
except Exception:
    CACHE_MAX_MB = 256
    CACHE_TTL_DAYS = 3650
    CACHE_ENABLE = True
    CACHE_MODE = "rw"



def _now() -> float:
    return time.time()

def _iso8601_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = _now()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _blake(data: bytes, n: int = 16) -> str:
    return hashlib.blake2b(data, digest_size=n).hexdigest()

def _json_dumps_sorted(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _safe_float3(xyz: Any) -> List[float]:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return [x, y, z]

def _norm_elem(e: str) -> str:
    if not e:
        return "C"
    return (e[0].upper() + e[1:].lower()) if len(e) > 1 else e.upper()



def _cov_radius(e: str) -> float:
    table = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
        "B": 0.85, "Si": 1.11, "Br": 1.20, "I": 1.39,
        "Na": 1.66, "K": 2.03, "Li": 1.28, "Mg": 1.41, "Ca": 1.76, "Al": 1.21, "Fe": 1.16,
        "Cu": 1.22, "Zn": 1.20, "Ag": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.44,
    }
    return table.get(_norm_elem(e), 0.75)

def _bond_guess_from_atoms(atoms: List[Dict[str, Any]], factor: float = 1.25) -> List[Tuple[int, int]]:
    n = len(atoms)
    out: List[Tuple[int, int]] = []
    for i in range(n - 1):
        ei = _norm_elem(atoms[i]["element"])
        ri = _cov_radius(ei)
        xi, yi, zi = atoms[i]["position"]
        for j in range(i + 1, n):
            ej = _norm_elem(atoms[j]["element"])
            rj = _cov_radius(ej)
            xj, yj, zj = atoms[j]["position"]
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d <= factor * (ri + rj):
                out.append((i, j))
    return out

def _formula_from_atoms(atoms: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for a in atoms:
        e = a.get("element")
        if e is None:
            e = a.get("e", "C")
        e = _norm_elem(e)
        counts[e] = counts.get(e, 0) + 1
    keys: List[str] = []
    if "C" in counts:
        keys.append("C")
    if "H" in counts:
        keys.append("H")
    keys += sorted(k for k in counts.keys() if k not in {"C", "H"})
    parts = []
    for k in keys:
        n = counts[k]
        parts.append(k if n == 1 else f"{k}{n}")
    return "".join(parts)

def _graph_signature_from_atoms_bonds(
    atoms: List[Dict[str, Any]],
    bonds: List[Tuple[int, int]],
) -> bytes:
    elems = [_norm_elem(a.get("element", a.get("e", "C"))) for a in atoms]
    deg = [0] * len(elems)
    for i, j in bonds:
        if 0 <= i < len(elems) and 0 <= j < len(elems) and i != j:
            deg[i] += 1
            deg[j] += 1
    order = sorted(range(len(elems)), key=lambda k: (elems[k], deg[k], k))
    pos = {old: new for new, old in enumerate(order)}
    remapped = []
    for i, j in bonds:
        if i == j:
            continue
        a, b = pos.get(i), pos.get(j)
        if a is None or b is None:
            continue
        if a > b:
            a, b = b, a
        remapped.append((a, b))
    remapped.sort()
    s = "|".join(f"{elems[i]}:{deg[i]}" for i in order) + "#" + ",".join(f"{i}-{j}" for i, j in remapped)
    return s.encode("utf-8")

@dataclass
class CacheEntryIndex:
    offset: int
    size: int
    last_used: float
    created: float
    atoms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CacheEntryIndex":
        return CacheEntryIndex(
            offset=int(d["offset"]),
            size=int(d["size"]),
            last_used=float(d.get("last_used", _now())),
            created=float(d.get("created", _now())),
            atoms=int(d.get("atoms", 0)),
        )

class Cache:
    def __init__(self, path: str = "./cache/molecules.jsonl", idx_path: Optional[str] = None,
                 max_mb: Optional[int] = None, ttl_days: Optional[int] = None) -> None:
        self.path = os.path.abspath(path)
        self.idx_path = os.path.abspath(idx_path) if idx_path else self.path + ".idx"
        self.max_bytes = int((max_mb if max_mb is not None else CACHE_MAX_MB) * 1024 * 1024)
        self.ttl_days = int(ttl_days if ttl_days is not None else CACHE_TTL_DAYS)
        _ensure_dir(self.path)
        self._lock = threading.RLock()
        self._index: Dict[str, CacheEntryIndex] = {}
        self._touch_count = 0
        self._load_index()


    def make_key(
        self,
        *,
        backend: str,
        meta: Dict[str, Any],
        method: str,
        scope: str,
        atoms: List[Dict[str, Any]],
        bonds: Optional[List] = None,
    ) -> str:
        be = str(backend or "unknown").lower()
        mth = str(method or "minimize").lower()
        sc = str(scope or "graph").lower()

        atoms_norm = self._normalize_atoms_for_key(atoms)

        meta_safe = dict(meta or {})
        meta_safe["backend"] = be
        meta_fp = _blake(_json_dumps_sorted(meta_safe).encode("utf-8"), n=8)

        if sc == "graph":
            if bonds is None:
                bonds = _bond_guess_from_atoms(atoms_norm, factor=1.25)
            bonds_norm: List[Tuple[int, int]] = []
            for p in bonds or []:
                try:
                    i, j = int(p[0]), int(p[1])
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    bonds_norm.append((i, j))
                except Exception:
                    continue
            gsig = _graph_signature_from_atoms_bonds(atoms_norm, bonds_norm)
            h = _blake(gsig, n=16)
            return f"{be}|{mth}|{meta_fp}|graph:{h}"

        f = _formula_from_atoms(atoms_norm)
        h = _blake(f.encode("utf-8"), n=12)
        return f"{be}|{mth}|{meta_fp}|formula:{h}"

    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            idx = self._index.get(key)
            if not idx:
                return None
            try:
                rec = self._read_raw(idx.offset, idx.size)
            except Exception:
                return None
            idx.last_used = _now()
            self._touch_count += 1
            if self._touch_count % 64 == 0:
                self._persist_index()
            return rec

    def store(self, key: str, rec: Dict[str, Any]) -> None:
        if "atoms" not in rec:
            raise ValueError("cache record missing 'atoms'")
        if "bonds" not in rec:
            rec["bonds"] = []
        if "ts" not in rec:
            rec["ts"] = _iso8601_utc()

        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            with open(self.path, "ab") as f:
                off = f.tell()
                f.write(line)

            atoms_n = len(rec.get("atoms") or [])
            self._index[key] = CacheEntryIndex(
                offset=off, size=len(line),
                last_used=_now(), created=_now(),
                atoms=atoms_n
            )
            self._persist_index()

            try:
                self._maybe_compact()
            except Exception:
                pass

    def record_from_result(
        self,
        result: Dict[str, Any],
        *,
        meta: Dict[str, Any],
        method: str,
        scope: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        atoms_in = result.get("atoms") or []
        atoms_compact = []
        for a in atoms_in:
            e = _norm_elem(a.get("element", a.get("e", "C")))
            pos = a.get("position", a.get("pos_A", [0.0, 0.0, 0.0]))
            atoms_compact.append({"e": e, "pos_A": _safe_float3(pos)})

        bonds = []
        rh = result.get("render_hints") or {}
        bsrc = rh.get("bonds") or result.get("bonds") or []
        for p in bsrc:
            try:
                i, j = int(p[0]), int(p[1])
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                bonds.append([i, j])
            except Exception:
                continue

        rec = {
            "atoms": atoms_compact,
            "bonds": bonds,
            "radii_A": result.get("atom_radii_A"),
            "identity": result.get("identity"),
            "meta": {
                **(meta or {}),
                "method": str(method or "minimize"),
                "scope": str(scope or "graph"),
            },
            "tags": list(tags or []),
            "ts": _iso8601_utc(),
        }
        return rec

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._index)
            size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
            return {
                "entries": total,
                "file_bytes": size,
                "max_bytes": self.max_bytes,
                "idx_bytes": os.path.getsize(self.idx_path) if os.path.exists(self.idx_path) else 0,
            }

    def close(self) -> None:
        with self._lock:
            self._persist_index()


    def _normalize_atoms_for_key(self, atoms_any: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for a in atoms_any or []:
            e = a.get("element")
            if e is None:
                e = a.get("e", "C")
            pos = a.get("position")
            if pos is None:
                pos = a.get("pos_A", [0.0, 0.0, 0.0])
            out.append({"element": _norm_elem(e), "position": _safe_float3(pos)})
        return out

    def _read_raw(self, offset: int, size: int) -> Dict[str, Any]:
        with open(self.path, "rb") as f:
            f.seek(offset)
            blob = f.read(size)
        return json.loads(blob)

    def _persist_index(self) -> None:
        tmp = self.idx_path + ".tmp"
        data = {k: v.to_dict() for k, v in self._index.items()}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.idx_path)

    def _load_index(self) -> None:
        if os.path.exists(self.idx_path):
            try:
                with open(self.idx_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._index = {k: CacheEntryIndex.from_dict(v) for k, v in raw.items()}
                return
            except Exception:
                self._index = {}
        if os.path.exists(self.path):
            try:
                self._rebuild_index_from_store()
                self._persist_index()
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _rebuild_index_from_store(self) -> None:
        self._index = {}
        offset = 0
        with open(self.path, "rb") as f:
            for line in f:
                size = len(line)
                try:
                    rec = json.loads(line)
                    meta = rec.get("meta") or {}
                    backend = str(meta.get("backend", "unknown"))
                    method = str(meta.get("method", "minimize"))
                    scope = str(meta.get("scope", "graph"))
                    atoms = rec.get("atoms") or []
                    bonds = rec.get("bonds") or []
                    key = self.make_key(
                        backend=backend, meta=meta, method=method, scope=scope,
                        atoms=atoms, bonds=bonds
                    )
                    self._index[key] = CacheEntryIndex(
                        offset=offset,
                        size=size,
                        last_used=_now(),
                        created=self._parse_ts(rec.get("ts")) or _now(),
                        atoms=len(atoms),
                    )
                except Exception:
                    pass
                offset += size

    def _parse_ts(self, ts: Any) -> Optional[float]:
        if not isinstance(ts, str):
            return None
        if not ts.endswith("Z") or "T" not in ts:
            return None
        try:
            import datetime
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except Exception:
            return None

    def _maybe_compact(self) -> None:
        if self.max_bytes <= 0:
            return
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if size <= self.max_bytes:
            return

        now = _now()
        ttl_sec = max(0, int(self.ttl_days * 86400)) if self.ttl_days else 0

        keys = list(self._index.keys())
        if ttl_sec > 0:
            expired = []
            for k, ent in list(self._index.items()):
                age = now - ent.created
                if age > ttl_sec:
                    expired.append(k)
            for k in expired:
                self._index.pop(k, None)

        target = int(self.max_bytes * 0.90)
        keep = sorted(self._index.items(), key=lambda kv: kv[1].last_used, reverse=True)

        tmp_path = self.path + ".compact.tmp"
        tmp_idx: Dict[str, CacheEntryIndex] = {}
        written = 0

        with open(tmp_path, "wb") as out:
            for k, ent in keep:
                try:
                    rec = self._read_raw(ent.offset, ent.size)
                except Exception:
                    continue
                line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
                if written + len(line) > target and written > 0:
                    break
                off = out.tell()
                out.write(line)
                tmp_idx[k] = CacheEntryIndex(
                    offset=off, size=len(line),
                    last_used=ent.last_used, created=ent.created, atoms=ent.atoms
                )
                written += len(line)

        os.replace(tmp_path, self.path)
        self._index = tmp_idx
        self._persist_index()
