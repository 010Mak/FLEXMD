from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, Response, send_from_directory

from utilities.config import (
    RUN_HOST, RUN_PORT, DEBUG, DEFAULT_BACKEND, DEFAULT_TIMESTEP_PS, FORCEFIELD_DIR,
    BACKENDS, MAX_ATOMS, MAX_STEPS, MAX_REPORT_FRAMES, MAX_REQUEST_BYTES
)
from utilities.radii import vdw_radius
from utilities.identify import identify_from_atoms

from simulation.system import System
from simulation.integrators import VerletIntegrator
from simulation.thermostats import LangevinThermostat
from simulation.engine_core import EngineCore

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

@app.before_request
def _limit_size():
    if request.path in ("/simulate", "/identify"):
        cl = request.content_length
        if cl is not None and cl > MAX_REQUEST_BYTES:
            return _error("request too large", 413)

@app.get("/health")
def health() -> Response:
    return jsonify(status="ok"), 200

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
    name = name.lower()
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
    ident = identify_from_atoms(atoms_json, allow_online=allow_online)
    return jsonify(status="success", identity=ident), 200

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
    if backend == "reaxff" and dt_ps > 0.001:
        return _error("reaxff timestep too large: use <= 0.001 ps (e.g., 0.00025 ps)", 400)

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

    system = System.from_json(atoms_json)

    integrator = VerletIntegrator(timestep=dt_ps)

    thermostat = None
    if str(payload.get("thermostat", "")).lower() == "langevin":
        default_temp = float(payload.get("defaultConditions", {}).get("temperature", 298.0))
        friction = float(payload.get("friction_coeff", 1.0))
        thermostat = LangevinThermostat(target_temp=default_temp, friction=friction)

    plugin_args = dict(payload.get("plugin_args", {}))

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
        results = engine.run(n_steps)
    except Exception:
        log.exception("simulation failed")
        return _error("simulation failed", 500)

    frames = []
    for r in results[::report_stride]:
        frames.append(
            {
                "step": r.step,
                "time_ps": r.time,
                "positions": r.positions.tolist(),
                "velocities": r.velocities.tolist(),
                "energy": r.energy,
                "forces": r.forces.tolist(),
            }
        )

    atoms_out = [{"element": a["element"]} for a in atoms_json]
    radii = [float(vdw_radius(a["element"])) for a in atoms_json]

    meta = {
        "backend_requested": backend,
        "selected_backend": getattr(engine.plugin, "NAME", "unknown"),
        "n_atoms": len(atoms_json),
        "n_steps": n_steps,
        "report_stride": report_stride,
        "report_stride_was_adjusted": stride_adjusted,
        "timestep_ps": dt_ps,
        "units": {"length": "angstrom", "time": "ps", "energy": "kcal/mol", "force": "kcal/mol/angstrom"},
    }

    include_identity = _as_bool(payload.get("include_identity"), True)
    allow_online_names = _as_bool(payload.get("allow_online_names"), False)
    identity = None
    if include_identity:
        try:
            if frames and "positions" in frames[0]:
                pos0 = frames[0]["positions"]
                id_atoms = [
                    {
                        "element": atoms_json[i]["element"],
                        "position": pos0[i],
                        "properties": (atoms_json[i].get("properties") or {}),
                    }
                    for i in range(len(atoms_json))
                ]
            else:
                id_atoms = [
                    {
                        "element": a["element"],
                        "position": a["position"],
                        "properties": (a.get("properties") or {}),
                    }
                    for a in atoms_json
                ]
            identity = identify_from_atoms(id_atoms, allow_online=allow_online_names)
        except Exception as e:
            log.warning("identity failed: %s", e)
            identity = None

    resp: Dict[str, Any] = {
        "status": "success",
        "meta": meta,
        "atoms": atoms_out,
        "atom_radii_A": radii,
        "trajectory": frames,
    }
    if identity:
        resp["identity"] = identity

    return jsonify(resp), 200

if __name__ == "__main__":
    FORCEFIELD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=RUN_HOST, port=RUN_PORT, debug=DEBUG)
