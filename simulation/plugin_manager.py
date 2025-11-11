from __future__ import annotations
import importlib
import logging
import pathlib

from simulation.plugin_interface import ForceCalculator

try:
    from importlib.metadata import entry_points
except Exception:  
    from importlib_metadata import entry_points 

log = logging.getLogger(__name__)

PLUGINS: dict[str, type[ForceCalculator]] = {}

def _load_entrypoint_plugins() -> None:
    try:
        eps = entry_points(group="mmdfled.plugins") 
    except TypeError:
        eps = entry_points().get("mmdfled.plugins", [])  
    for ep in eps:
        try:
            cls = ep.load()
            if (
                isinstance(cls, type)
                and issubclass(cls, ForceCalculator)
                and cls is not ForceCalculator
            ):
                PLUGINS[cls.NAME] = cls
            else:
                log.warning("entry point %s does not implement ForceCalculator", getattr(ep, "name", ep))
        except Exception as e:  
            log.warning("failed to load plugin entry point %s: %s", getattr(ep, "name", ep), e)

def _load_local_plugins() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    plugins_dir = root / "plugins"
    if not plugins_dir.is_dir():
        return
    for path in plugins_dir.glob("*.py"):
        if path.stem == "__init__":
            continue
        module_name = f"plugins.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            log.warning("skipping local plugin %s: %s", module_name, e)
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, ForceCalculator)
                and obj is not ForceCalculator
            ):
                PLUGINS[obj.NAME] = obj

_load_entrypoint_plugins()
_load_local_plugins()

if not PLUGINS:
    raise RuntimeError("no ForceCalculator plugins found; check installation or local plugins/")

__all__ = ["PLUGINS"]
