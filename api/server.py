from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import time
import datetime

from flask import Flask, jsonify, request, Response, send_from_directory

from utilities.config import (
    RUN_HOST,
    RUN_PORT,
    DEBUG,
    DEFAULT_BACKEND,
    DEFAULT_TIMESTEP_PS,
    BACKENDS,
    MAX_ATOMS,
    MAX_STEPS,
    MAX_REPORT_FRAMES,
    MAX_REQUEST_BYTES,
    DISCORD_WEBHOOK_URL,
    WEBHOOK_ON_STARTUP,
    WEBHOOK_ON_SIMULATE,
    SERVER_NAME,
    SERVER_LOCATION,
    REAXFF_MAX_DT_PS,
)

try:
    from utilities.config import (
        CACHE_ENABLE as _CFG_CACHE_ENABLE,
        CACHE_MODE as _CFG_CACHE_MODE,
        CACHE_PATH as _CFG_CACHE_PATH,
        CACHE_BACKENDS as _CFG_CACHE_BACKENDS,
        CACHE_MAX_ATOMS as _CFG_CACHE_MAX_ATOMS,
    )
    _HAS_CACHE_CFG = True
except Exception:
    _HAS_CACHE_CFG = False
    _CFG_CACHE_ENABLE = True
    _CFG_CACHE_MODE = "rw"
    _CFG_CACHE_PATH = "./cache/molecules.jsonl"
    _CFG_CACHE_BACKENDS = ["smirnoff", "reaxff"]
    _CFG_CACHE_MAX_ATOMS = 128

try:
    from utilities.cache import Cache
    _CACHE: Optional[Cache] = Cache(path=_CFG_CACHE_PATH)
    _CACHE_AVAILABLE = True
except Exception:
    _CACHE = None
    _CACHE_AVAILABLE = False

from utilities.radii import vdw_radius
from utilities.identify import identify_from_atoms
from utilities.ID import compute_all_ids

from simulation.system import System
from simulation.integrators import VerletIntegrator
from simulation.thermostats import LangevinThermostat
from simulation.engine_core import EngineCore

from utilities import status as status_util
from utilities import discord_webhook as dwh

from utilities.ghs import ghs_from_inchikey
from utilities.physprops import physprops_from_inchikey, build_phase_curve_1atm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("server")

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"

app = Flask(__name__)


def _error(msg: str, code: int = 500):
    log.error(msg)
    return jsonify(status="error", message=msg), code


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def _no_store(resp: Response) -> Response:
    resp.cache_control.no_store = True
    resp.cache_control.max_age = 0
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _validate_atoms(items: Any) -> Tuple[bool, str]:
    if not isinstance(items, list) or not items:
        return False, "'atoms' must be a non-empty list"
    for i, a in enumerate(items):
        if not isinstance(a, dict):
            return False, f"atom #{i} must be an object"
        el = a.get("element", None)
        pos = a.get("position", None)
        if not isinstance(el, str) or not el:
            return False, f"atom #{i} missing 'element' string"
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            return False, f"atom #{i} position must be [x,y,z]"
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            return False, f"atom #{i} position must be numeric"
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False, f"atom #{i} position must be finite"
    return True, ""


def _normalize_cache_policy(global_mode: str, allow: bool, requested: str) -> str:
    if not allow:
        return "off"
    req = (requested or "auto").strip().lower()
    gm = (global_mode or "rw").strip().lower()
    if req in {"off", "read", "write", "rw"}:
        if gm == "off":
            return "off"
        if gm == "read" and req in {"rw", "write"}:
            return "read"
        if gm == "write" and req in {"rw", "read"}:
            return "write"
        return req
    return gm if gm in {"read", "write", "rw"} else "rw"


_COV_RAD = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "B": 0.85, "Si": 1.11, "Br": 1.20, "I": 1.39,
    "Na": 1.66, "K": 2.03, "Li": 1.28, "Mg": 1.41, "Ca": 1.76, "Al": 1.21, "Fe": 1.16,
    "Cu": 1.22, "Zn": 1.20, "Ag": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.44,
}
def _norm_elem(e: str) -> str:
    if not e:
        return "C"
    return (e[0].upper() + e[1:].lower()) if len(e) > 1 else e.upper()

def _bond_guess(atoms_json: List[Dict[str, Any]], factor: float = 1.25) -> List[Tuple[int, int]]:
    n = len(atoms_json)
    out: List[Tuple[int, int]] = []
    for i in range(n - 1):
        ei = _norm_elem(atoms_json[i]["element"])
        ri = _COV_RAD.get(ei, 0.75)
        xi, yi, zi = map(float, atoms_json[i]["position"])
        for j in range(i + 1, n):
            ej = _norm_elem(atoms_json[j]["element"])
            rj = _COV_RAD.get(ej, 0.75)
            xj, yj, zj = map(float, atoms_json[j]["position"])
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= factor * (ri + rj):
                out.append((i, j))
    return out


def _extract_formal_charges(atoms_json: List[Dict[str, Any]]) -> Optional[List[int]]:
    charges: List[int] = []
    any_nonzero = False
    for a in atoms_json:
        props = a.get("properties") or {}
        fc = int(props.get("formal_charge", 0) or 0)
        charges.append(fc)
        if fc != 0:
            any_nonzero = True
    return charges if any_nonzero else None


def _elements_from_atoms(atoms_json: List[Dict[str, Any]]) -> List[str]:
    return [_norm_elem(a["element"]) for a in atoms_json]

def _positions_from_atoms(atoms_json: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
    return [tuple(map(float, a["position"])) for a in atoms_json]


def _build_cached_response(
    *,
    backend: str,
    atoms_json: List[Dict[str, Any]],
    cache_rec: Dict[str, Any],
    include_thermo: bool,
    include_identity: bool,
    include_render_hints: bool,
    include_ghs: bool,
    stride_adjusted: bool,
    report_stride: int,
    dt_ps: float,
    n_steps: int,
    thermostat_name: Optional[str],
    default_tempK: Optional[float],
    plugin_args: Dict[str, Any],
    cache_scope: str,
    bonds_for_key: List[Tuple[int, int]],
) -> Dict[str, Any]:
    cached_atoms = cache_rec.get("atoms", []) or []
    posA = [a.get("position", a.get("pos_A", [0.0, 0.0, 0.0])) for a in cached_atoms]
    n = len(posA)
    zeros = [[0.0, 0.0, 0.0] for _ in range(n)]

    frame: Dict[str, Any] = {
        "step": n_steps,
        "time_ps": dt_ps * n_steps,
        "positions": posA,
        "velocities": zeros,
        "energy": None,
        "forces": zeros,
    }
    if include_thermo:
        pass

    atoms_out = [{"element": a.get("e") or _norm_elem(a.get("element", "C"))} for a in cached_atoms] \
                or [{"element": _norm_elem(a["element"])} for a in atoms_json]
    radii = [float(vdw_radius(a["element"])) for a in atoms_out]

    render_hints = None
    if include_render_hints:
        render_hints = {"bonds": cache_rec.get("bonds", [])}

    ident = None
    if include_identity:
        ident = cache_rec.get("identity", None)
        if ident and include_ghs:
            pass

    meta: Dict[str, Any] = {
        "backend_requested": backend,
        "selected_backend": backend,
        "n_atoms": len(atoms_out),
        "n_steps": n_steps,
        "report_stride": report_stride,
        "report_stride_was_adjusted": bool(stride_adjusted),
        "timestep_ps": dt_ps,
        "units": {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "length": "angstrom",
            "time": "ps",
        },
        "server": {"name": SERVER_NAME, "location": SERVER_LOCATION},
    }

    out = {
        "status": "success",
        "meta": meta,
        "atoms": atoms_out,
        "atom_radii_A": radii,
        "trajectory": [frame],
        "source": "cache",
        "cache": {
            "hit": True,
            "key": cache_rec.get("_key") or None,
            "policy": cache_rec.get("_policy") or None,
            "scope": (cache_rec.get("meta") or {}).get("scope", "graph"),
            "method": (cache_rec.get("meta") or {}).get("method", "minimize"),
            "age_seconds": cache_rec.get("_age_seconds"),
            "source": "cache",
        },
    }
    if ident is not None:
        out["identity"] = ident
    if render_hints is not None:
        out["render_hints"] = render_hints

    try:
        rec_gid = cache_rec.get("graph_id")
        rec_frags = cache_rec.get("fragments")
        rec_mixture = cache_rec.get("mixture_id")
        rec_ms = cache_rec.get("method_signature")
        rec_rid = cache_rec.get("result_id")

        elements = [a["element"] for a in atoms_out]
        pos_tuples = [tuple(map(float, p)) for p in posA]
        ids_pkg = None

        if not (rec_gid and rec_frags and rec_ms and rec_rid):
            ids_pkg = compute_all_ids(
                elements=elements,
                positionsA=pos_tuples,
                bonds0=bonds_for_key or (render_hints["bonds"] if render_hints else []),
                backend=backend,
                thermostat_name=thermostat_name,
                dt_ps=dt_ps,
                n_steps=n_steps,
                report_stride=report_stride,
                plugin_args=plugin_args or {},
                scope=cache_scope,
                default_tempK=default_tempK,
            )

        out["graph_id"] = rec_gid or (ids_pkg and ids_pkg["graph_id"])
        out["fragments"] = rec_frags or (ids_pkg and ids_pkg["fragments"]) or []
        out["fragment_count"] = len(out["fragments"])
        out["mixture_id"] = rec_mixture or (ids_pkg and ids_pkg.get("mixture_id"))
        out["method_signature"] = rec_ms or (ids_pkg and ids_pkg["method_signature"])
        out["result_id"] = cache_rec.get("_key") or rec_rid or (ids_pkg and ids_pkg["result_id"])
    except Exception as e:
        log.warning("ID computation (cache path) failed: %s", e)

    return out


@app.before_request
def _limit_size():
    if request.path in ("/simulate", "/identify", "/properties"):
        cl = request.content_length
        if cl is not None and cl > MAX_REQUEST_BYTES:
            return _error("request too large", 413)


_STARTUP_WEBHOOK_SENT = False


def _send_startup_webhook():
    try:
        if DISCORD_WEBHOOK_URL and WEBHOOK_ON_STARTUP:
            info = status_util.gather_server_status()
            embed = dwh.server_embed(info)
            dwh.post(DISCORD_WEBHOOK_URL, embeds=[embed], wait=True)
            log.info("startup webhook posted")
    except Exception as e:
        log.warning("startup webhook failed: %s", e)


if hasattr(app, "before_serving"):
    @app.before_serving
    def _startup_webhook_before_serving():
        global _STARTUP_WEBHOOK_SENT
        if not _STARTUP_WEBHOOK_SENT:
            _send_startup_webhook()
            _STARTUP_WEBHOOK_SENT = True
else:
    @app.before_request
    def _startup_webhook_fallback():
        global _STARTUP_WEBHOOK_SENT
        if not _STARTUP_WEBHOOK_SENT:
            _send_startup_webhook()
            _STARTUP_WEBHOOK_SENT = True


@app.get("/health")
def health() -> Response:
    return jsonify(status="ok"), 200


@app.get("/status")
def status() -> Response:
    try:
        info = status_util.gather_server_status()
        return jsonify(
            {
                "status": "ok",
                "server": info["server"],
                "config": info["config"],
                "plugins": info["plugins"],
                "smirnoff": info["smirnoff"],
                "reaxff": info["reaxff"],
            }
        ), 200
    except Exception as e:
        log.exception("status failed")
        return _error(f"status failed: {e}", 500)


@app.get("/demo")
def demo_index() -> Response:
    index_path = DEMO_DIR / "index.html"
    if not index_path.exists():
        return _error("demo/index.html not found", 404)
    resp = send_from_directory(str(DEMO_DIR), "index.html")
    return _no_store(resp)


@app.get("/demo/examples")
def demo_examples() -> Response:
    resp = jsonify(["methane", "water", "hydroxide"])
    return _no_store(resp)


@app.get("/demo/examples/<name>")
def demo_example(name: str) -> Response:
    name = (name or "").lower()
    if name == "methane":
        atoms = [
            {"element": "C", "position": [0.0, 0.0, 0.0]},
            {"element": "H", "position": [0.629, 0.629, 0.629]},
            {"element": "H", "position": [0.629, -0.629, -0.629]},
            {"element": "H", "position": [-0.629, 0.629, -0.629]},
            {"element": "H", "position": [-0.629, -0.629, 0.629]},
        ]
    elif name == "hydroxide":
        atoms = [
            {"element": "O", "position": [0.000, 0.000, 0.000], "properties": {"formal_charge": -1}},
            {"element": "H", "position": [0.970, 0.000, 0.000]},
        ]
    else:
        atoms = [
            {"element": "O", "position": [0.000, 0.000, 0.000]},
            {"element": "H", "position": [0.9572, 0.000, 0.000]},
            {"element": "H", "position": [-0.2399872, 0.927297, 0.000]},
        ]
    resp = jsonify({"atoms": atoms})
    return _no_store(resp)


@app.get("/demo/<path:filename>")
def demo_files(filename: str) -> Response:
    resp = send_from_directory(str(DEMO_DIR), filename)
    return _no_store(resp)


@app.post("/identify")
def identify() -> Response:
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return _error(f"bad json: {e}", 400)

    atoms_json = payload.get("atoms") or []
    ok, msg = _validate_atoms(atoms_json)
    if not ok:
        return _error(msg, 400)

    allow_online = _as_bool(payload.get("allow_online_names"), False)
    include_ghs = _as_bool(payload.get("include_ghs"), False)

    ident = identify_from_atoms(atoms_json, allow_online=allow_online)

    if include_ghs:
        ik = (ident or {}).get("inchikey")
        try:
            ghs = ghs_from_inchikey(ik) if ik else {"hazard_meanings": [], "hazard_label": ""}
        except Exception as e:
            log.warning("GHS lookup failed: %s", e)
            ghs = {"hazard_meanings": [], "hazard_label": ""}

        ident["ghs_pictograms"] = {
            "hazard_meanings": ghs["hazard_meanings"],
            "hazard_label": ghs["hazard_label"],
            "source": "PubChem GHS",
        }
        ident["ghs_pictogram_names"] = ghs["hazard_label"]

    return jsonify(status="success", identity=ident), 200


@app.post("/properties")
def properties() -> Response:
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return _error(f"bad json: {e}", 400)

    atoms_json: List[Dict[str, Any]] = payload.get("atoms") or []
    ok, msg = _validate_atoms(atoms_json)
    if not ok:
        return _error(msg, 400)

    allow_online = _as_bool(payload.get("allow_online_names"), False)

    try:
        phase_resolution = int(payload.get("phase_resolution", 200))
    except Exception:
        phase_resolution = 200
    if phase_resolution < 3:
        phase_resolution = 3

    t_min_req = payload.get("t_min_K")
    t_max_req = payload.get("t_max_K")
    t_min_K = float(t_min_req) if t_min_req is not None else None
    t_max_K = float(t_max_req) if t_max_req is not None else None

    ident: Optional[Dict[str, Any]] = None
    try:
        ident = identify_from_atoms(atoms_json, allow_online=allow_online)
    except Exception as e:
        log.warning("identity failed in /properties: %s", e)
        ident = None

    props: Dict[str, Any] = {}
    mpK: Optional[float] = None
    bpK: Optional[float] = None
    try:
        ik = (ident or {}).get("inchikey")
        if ik:
            props = physprops_from_inchikey(ik)
            mp = props.get("melting_point")
            bp = props.get("boiling_point")
            if mp and isinstance(mp.get("value_K"), (int, float)):
                mpK = float(mp["value_K"])
            if bp and isinstance(bp.get("value_K"), (int, float)):
                bpK = float(bp["value_K"])
    except Exception as e:
        log.warning("physprops lookup failed in /properties: %s", e)

    phase_diagram: Optional[Dict[str, Any]] = None
    if mpK is not None and bpK is not None and bpK > mpK:
        phase_diagram = build_phase_curve_1atm(
            melting_point_K=mpK,
            boiling_point_K=bpK,
            t_min_K=t_min_K,
            t_max_K=t_max_K,
            n_points=phase_resolution,
        )

    out: Dict[str, Any] = {
        "status": "success",
        "identity": ident,
        "properties": props,
    }
    if phase_diagram is not None:
        out["phase_diagram"] = phase_diagram

    return jsonify(out), 200


@app.post("/simulate")
def simulate() -> Response:
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return _error(f"bad json: {e}", 400)

    atoms_json: List[Dict[str, Any]] = payload.get("atoms") or []
    ok, msg = _validate_atoms(atoms_json)
    if not ok:
        return _error(msg, 400)
    if len(atoms_json) > MAX_ATOMS:
        return _error(f"too many atoms: {len(atoms_json)} > {MAX_ATOMS}", 400)

    backend = str(payload.get("backend", DEFAULT_BACKEND)).lower()
    if backend not in BACKENDS:
        return _error(f"invalid backend '{backend}'", 400)

    dt_ps = float(payload.get("timestep_ps", DEFAULT_TIMESTEP_PS))
    if backend == "reaxff" and dt_ps > REAXFF_MAX_DT_PS:
        return _error(f"reaxff timestep too large: use <= {REAXFF_MAX_DT_PS} ps (e.g., 0.00025 ps)", 400)

    n_steps = int(payload.get("n_steps", 1))
    if n_steps < 1 or n_steps > MAX_STEPS:
        return _error(f"n_steps out of range: 1..{MAX_STEPS}", 400)

    report_stride_req = max(1, int(payload.get("report_stride", 1)))
    frames_budget = (n_steps + report_stride_req - 1) // report_stride_req
    report_stride = report_stride_req
    stride_adjusted = False
    if frames_budget > MAX_REPORT_FRAMES:
        report_stride = math.ceil(n_steps / MAX_REPORT_FRAMES)
        stride_adjusted = True

    include_thermo = _as_bool(payload.get("include_thermo"), False)
    include_identity = _as_bool(payload.get("include_identity"), False)
    allow_online_names = _as_bool(payload.get("allow_online_names"), False)
    include_render_hints = _as_bool(payload.get("include_render_hints"), False)
    include_ghs = _as_bool(payload.get("include_ghs"), False)

    allow_cache = _as_bool(payload.get("allow_cache", True), True)
    cache_policy_req = str(payload.get("cache_policy", "auto")).lower()
    cache_scope = str(payload.get("cache_scope", "graph")).lower()
    cache_method = str(payload.get("cache_method", "minimize")).lower()
    cache_tags = payload.get("cache_tags", []) or []
    canonical_minimize = _as_bool(payload.get("canonical_minimize", False), False)
    bonds_hint = payload.get("bonds") or (payload.get("render_hints") or {}).get("bonds")

    thermostat_name = str(payload.get("thermostat", "")).lower() or None
    default_temp = None
    if thermostat_name == "langevin":
        default_temp = float(payload.get("defaultConditions", {}).get("temperature", 298.0))
    plugin_args = dict(payload.get("plugin_args", {}) or {})

    cache_hit = False
    cache_key_used: Optional[str] = None
    cached_response: Optional[Dict[str, Any]] = None

    cache_globally_enabled = bool(_CACHE_AVAILABLE and _CFG_CACHE_ENABLE)
    policy = _normalize_cache_policy(_CFG_CACHE_MODE, allow_cache, cache_policy_req)
    cache_backend_allowed = backend in set(_CFG_CACHE_BACKENDS or [])

    bonds_for_key: List[Tuple[int, int]] = []
    if isinstance(bonds_hint, list) and bonds_hint:
        bonds_for_key = [(int(i), int(j)) for (i, j) in bonds_hint]
    else:
        bonds_for_key = _bond_guess(atoms_json, factor=1.25)

    if cache_globally_enabled and cache_backend_allowed and policy in {"read", "rw"}:
        try:
            key = _CACHE.make_key(
                backend=backend,
                meta={"backend": backend},
                method=cache_method or "minimize",
                scope=cache_scope or "graph",
                atoms=atoms_json,
                bonds=bonds_for_key,
            )
            rec = _CACHE.lookup(key)
            if rec:
                rec["_key"] = key
                rec["_policy"] = policy
                age_sec: Optional[float] = None
                ts = rec.get("ts")
                if isinstance(ts, str) and ts.endswith("Z") and "T" in ts:
                    try:
                        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                        age_sec = (datetime.datetime.utcnow() - dt).total_seconds()
                    except Exception:
                        age_sec = None
                rec["_age_seconds"] = age_sec

                cached_response = _build_cached_response(
                    backend=backend,
                    atoms_json=atoms_json,
                    cache_rec=rec,
                    include_thermo=include_thermo,
                    include_identity=include_identity,
                    include_render_hints=include_render_hints,
                    include_ghs=include_ghs,
                    stride_adjusted=stride_adjusted,
                    report_stride=report_stride,
                    dt_ps=dt_ps,
                    n_steps=n_steps,
                    thermostat_name=thermostat_name,
                    default_tempK=default_temp,
                    plugin_args=plugin_args,
                    cache_scope=cache_scope,
                    bonds_for_key=bonds_for_key,
                )
                cache_hit = True
                cache_key_used = key
        except Exception as e:
            log.warning("cache lookup failed (non-fatal): %s", e)

    if cache_hit and cached_response:
        return jsonify(cached_response), 200

    system = System.from_json(atoms_json)
    integrator = VerletIntegrator(timestep=dt_ps)

    thermostat = None
    if thermostat_name == "langevin":
        friction = float(payload.get("friction_coeff", 1.0))
        thermostat = LangevinThermostat(target_temp=float(default_temp or 298.0), friction=friction)

    try:
        engine = EngineCore.from_config(
            system=system,
            integrator=integrator,
            system_kwargs={
                "backend": backend,
                "plugin_args": plugin_args,
                "thermostat": thermostat,
            },
        )
    except Exception as e:
        return _error(f"engine setup failed: {e}", 400)

    try:
        results = engine.run(n_steps, include_thermo=include_thermo, report_stride=report_stride)
    except Exception:
        log.exception("simulation failed")
        return _error("simulation failed", 500)

    frames: List[Dict[str, Any]] = []
    for r in results:
        frames.append(r.to_dict())

    atoms_out = [{"element": _norm_elem(a["element"])} for a in atoms_json]
    radii = [float(vdw_radius(a["element"])) for a in atoms_json]

    ident = None
    if include_identity:
        try:
            ident = identify_from_atoms(atoms_json, allow_online=allow_online_names)
        except Exception as e:
            log.warning("identity failed: %s", e)
            ident = None

        if ident and include_ghs:
            ik = (ident or {}).get("inchikey")
            try:
                ghs = ghs_from_inchikey(ik) if ik else {"hazard_meanings": [], "hazard_label": ""}
            except Exception as e:
                log.warning("GHS lookup failed: %s", e)
                ghs = {"hazard_meanings": [], "hazard_label": ""}

            ident.setdefault("ghs_pictograms", {})
            ident["ghs_pictograms"].update(
                {
                    "hazard_meanings": ghs["hazard_meanings"],
                    "hazard_label": ghs["hazard_label"],
                    "source": "PubChem GHS",
                }
            )
            ident["ghs_pictogram_names"] = ghs["hazard_label"]

    render_hints = None
    if include_render_hints:
        try:
            rh = getattr(engine.plugin, "render_hints", None)
            render_hints = rh() if callable(rh) else None
        except Exception:
            render_hints = None

    meta: Dict[str, Any] = {
        "backend_requested": backend,
        "selected_backend": getattr(engine.plugin, "NAME", "unknown"),
        "n_atoms": len(atoms_json),
        "n_steps": n_steps,
        "report_stride": report_stride,
        "report_stride_was_adjusted": bool(stride_adjusted),
        "timestep_ps": dt_ps,
        "units": {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "length": "angstrom",
            "time": "ps",
        },
        "server": {"name": SERVER_NAME, "location": SERVER_LOCATION},
    }

    out = {
        "status": "success",
        "meta": meta,
        "atoms": atoms_out,
        "atom_radii_A": radii,
        "trajectory": frames,
        "source": "backend",
    }
    if ident is not None:
        out["identity"] = ident
    if render_hints is not None:
        out["render_hints"] = render_hints

    out["cache"] = {
        "hit": False,
        "key": None,
        "policy": _normalize_cache_policy(_CFG_CACHE_MODE, allow_cache, cache_policy_req),
        "scope": cache_scope,
        "method": cache_method,
        "age_seconds": None,
        "source": "backend",
    }

    try:
        last = frames[-1] if frames else None
        last_positions = last["positions"] if last and isinstance(last.get("positions"), list) else []
        elements_list = [a["element"] for a in atoms_out]

        bonds_used = []
        if render_hints and isinstance(render_hints.get("bonds"), list) and render_hints["bonds"]:
            bonds_used = [(int(i), int(j)) for (i, j, *_) in render_hints["bonds"]]
        elif isinstance(bonds_hint, list) and bonds_hint:
            bonds_used = [(int(i), int(j)) for (i, j) in bonds_hint]
        else:
            bonds_used = _bond_guess([{"element": e, "position": p} for e, p in zip(elements_list, last_positions or _positions_from_atoms(atoms_json))], factor=1.25)

        ids_pkg = compute_all_ids(
            elements=elements_list,
            positionsA=[tuple(map(float, p)) for p in (last_positions or _positions_from_atoms(atoms_json))],
            bonds0=bonds_used,
            backend=meta["selected_backend"] or backend,
            thermostat_name=thermostat_name,
            dt_ps=dt_ps,
            n_steps=n_steps,
            report_stride=report_stride,
            plugin_args=plugin_args or {},
            scope=cache_scope,
            default_tempK=default_temp,
        )

        out["graph_id"] = ids_pkg["graph_id"]
        out["conformer_id"] = ids_pkg["conformer_id"]
        out["fragments"] = ids_pkg["fragments"]
        out["fragment_count"] = ids_pkg["fragment_count"]
        if ids_pkg.get("mixture_id"):
            out["mixture_id"] = ids_pkg.get("mixture_id")
        out["method_signature"] = ids_pkg["method_signature"]
        out["result_id"] = ids_pkg["result_id"]
    except Exception as e:
        log.warning("ID computation (fresh path) failed: %s", e)

    try:
        if (
            cache_globally_enabled
            and (backend in set(_CFG_CACHE_BACKENDS or []))
            and _CFG_CACHE_MAX_ATOMS >= len(atoms_json)
            and out.get("render_hints") and isinstance(out["render_hints"].get("bonds"), list) and out["render_hints"]["bonds"]
            and out["cache"]["policy"] in {"write", "rw"}
        ):
            last = frames[-1] if frames else None
            if last and isinstance(last.get("positions"), list):
                atoms_for_cache = [
                    {"element": a["element"], "position": last["positions"][i]}
                    for i, a in enumerate(atoms_out)
                ]
                result_for_cache = {
                    "atoms": atoms_for_cache,
                    "trajectory": [last],
                    "render_hints": {"bonds": out["render_hints"]["bonds"]},
                    "identity": out.get("identity"),
                    "atom_radii_A": out.get("atom_radii_A"),
                    "graph_id": out.get("graph_id"),
                    "conformer_id": out.get("conformer_id"),
                    "fragments": out.get("fragments"),
                    "fragment_count": out.get("fragment_count"),
                    "mixture_id": out.get("mixture_id"),
                    "method_signature": out.get("method_signature"),
                    "result_id": out.get("result_id"),
                }
                precise_backend = meta.get("selected_backend", backend) or backend
                store_key = _CACHE.make_key(
                    backend=precise_backend,
                    meta={"backend": precise_backend},
                    method=cache_method or ("minimize" if canonical_minimize else "relax+nve"),
                    scope=cache_scope or "graph",
                    atoms=atoms_for_cache,
                    bonds=out["render_hints"]["bonds"],
                )
                rec = _CACHE.record_from_result(
                    result_for_cache,
                    meta={"backend": precise_backend, "scope": cache_scope, "method": cache_method},
                    method=cache_method,
                    scope=cache_scope,
                    tags=cache_tags or [],
                )
                _CACHE.store(store_key, rec)
                out["cache"]["key"] = store_key
                out["result_id"] = store_key
    except Exception as e:
        log.warning("cache store skipped (non-fatal): %s", e)

    return jsonify(out), 200


if __name__ == "__main__":
    app.run(host=RUN_HOST, port=RUN_PORT, debug=DEBUG)
