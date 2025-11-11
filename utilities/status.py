from __future__ import annotations
from typing import Any, Dict
from utilities.config import (
    RUN_HOST, RUN_PORT, MAX_ATOMS, MAX_STEPS, DEFAULT_TEMPERATURE_K,
    SERVER_NAME, SERVER_LOCATION, FORCEFIELD_DIR
)
from utilities.ffield_utils import scan_potentials_dir
from simulation.plugin_manager import PLUGINS

def gather_server_status() -> Dict[str, Any]:
    plugins: Dict[str, Any] = {}
    for name, cls in PLUGINS.items():
        ok = False
        try:
            ok = bool(getattr(cls, "is_available", lambda: True)())
        except Exception:
            ok = False
        extra = {}
        for attr in ("DEFAULT_OFFXML","DEFAULT_FORCEFIELD","NAME","CAPABILITY"):
            if hasattr(cls, attr):
                extra[attr.lower()] = getattr(cls, attr)
        plugins[name] = {"available": ok, **extra}

    smirnoff_info = {}
    if "smirnoff" in PLUGINS:
        offxml = getattr(PLUGINS["smirnoff"], "DEFAULT_OFFXML", None)
        if offxml:
            smirnoff_info["offxml_id"] = offxml

    reaxff_info = scan_potentials_dir(str(FORCEFIELD_DIR))

    return {
        "server": {"name": SERVER_NAME, "location": SERVER_LOCATION},
        "config": {
            "host": RUN_HOST, "port": RUN_PORT,
            "max_atoms": MAX_ATOMS, "max_steps": MAX_STEPS,
            "default_temperature_K": DEFAULT_TEMPERATURE_K
        },
        "plugins": plugins,
        "smirnoff": smirnoff_info,
        "reaxff": reaxff_info
    }
