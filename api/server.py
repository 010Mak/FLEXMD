from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from utilities.radii import vdw_radius
from utilities.identify import identify_from_atoms

from simulation.system import System
from simulation.integrators import VerletIntegrator
from simulation.thermostats import LangevinThermostat
from simulation.engine_core import EngineCore

from utilities import status as status_util
from utilities import discord_webhook as dwh

from utilities.ghs import ghs_from_inchikey

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
        results = engine.run(n_steps, include_thermo=include_thermo)
    except Exception:
        log.exception("simulation failed")
        return _error("simulation failed", 500)

    frames: List[Dict[str, Any]] = []
    for r in results[::report_stride]:
        frame: Dict[str, Any] = {
            "step": r.step,
            "time_ps": r.time,
            "positions": r.positions.tolist(),
            "velocities": r.velocities.tolist(),
            "energy": r.energy,
            "forces": r.forces.tolist(),
        }
        if include_thermo:
            ke = getattr(r, "kinetic", None)
            tK = getattr(r, "temperature", None)
            te = getattr(r, "total_energy", None)
            if ke is not None:
                frame["kinetic_energy_kcal_per_mol"] = float(ke)
            if tK is not None:
                frame["temperature_K"] = float(tK)
            if te is not None:
                frame["total_energy_kcal_per_mol"] = float(te)
        frames.append(frame)

    atoms_out = [{"element": a["element"]} for a in atoms_json]
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
    }
    if ident is not None:
        out["identity"] = ident
    if render_hints is not None:
        out["render_hints"] = render_hints

    return jsonify(out), 200
