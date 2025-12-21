from __future__ import annotations

import datetime
import io
import json
import uuid
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional


DEFAULT_USER_AGENT = "FLEXMD/1.0 (+https://github.com/010Mak/FLEXMD)"

def _valid_discord_webhook(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url or "")
        parts = [x for x in p.path.split("/") if x]
        return (
            p.scheme in ("http", "https")
            and (p.netloc.endswith("discord.com") or p.netloc.endswith("discordapp.com"))
            and len(parts) >= 4
            and parts[0] == "api"
            and parts[1] == "webhooks"
        )
    except Exception:
        return False

def _iso_utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _append_query(url: str, **params: Any) -> str:
    p = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(p.query, keep_blank_values=True))
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            q[k] = "true" if v else "false"
        else:
            q[k] = str(v)
    new_query = urllib.parse.urlencode(q)
    return urllib.parse.urlunparse(p._replace(query=new_query))


def post(
    url: str,
    *,
    content: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
    wait: bool = True,
    username: Optional[str] = None,
    timeout_s: int = 10,
    raise_on_error: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> None:
    if not _valid_discord_webhook(url):
        if raise_on_error:
            raise ValueError("invalid discord webhook URL")
        print("[discord] invalid webhook URL; skipping")
        return

    url2 = _append_query(url, wait=True) if wait else url

    payload: Dict[str, Any] = {}
    if content:
        payload["content"] = str(content)[:1900]
    if embeds:
        payload["embeds"] = embeds[:10]
    if username:
        payload["username"] = username

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url2,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read(256)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        if raise_on_error:
            raise RuntimeError(f"discord webhook HTTPError {e.code}: {msg}") from e
        print(f"[discord] HTTPError {e.code}: {msg}")
    except Exception as e:
        if raise_on_error:
            raise
        print(f"[discord] error posting webhook: {e}")


def post_file(
    url: str,
    *,
    filename: str,
    file_bytes: bytes,
    content: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
    wait: bool = True,
    username: Optional[str] = None,
    content_type: str = "application/octet-stream",
    timeout_s: int = 20,
    raise_on_error: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> None:
    if not _valid_discord_webhook(url):
        if raise_on_error:
            raise ValueError("invalid discord webhook URL")
        print("[discord] invalid webhook URL; skipping")
        return

    url2 = _append_query(url, wait=True) if wait else url

    payload: Dict[str, Any] = {}
    if content:
        payload["content"] = str(content)[:1900]
    if embeds:
        payload["embeds"] = embeds[:10]
    if username:
        payload["username"] = username

    boundary = "----flexmd-" + uuid.uuid4().hex
    b = boundary.encode("utf-8")

    bio = io.BytesIO()

    def _w(x: bytes) -> None:
        bio.write(x)

    _w(b"--" + b + b"\r\n")
    _w(b'Content-Disposition: form-data; name="payload_json"\r\n')
    _w(b"Content-Type: application/json\r\n\r\n")
    _w(json.dumps(payload).encode("utf-8"))
    _w(b"\r\n")

    safe_fn = (filename or "file.bin").replace('"', "_")
    _w(b"--" + b + b"\r\n")
    _w(f'Content-Disposition: form-data; name="files[0]"; filename="{safe_fn}"\r\n'.encode("utf-8"))
    _w(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    _w(file_bytes)
    _w(b"\r\n")

    _w(b"--" + b + b"--\r\n")

    data = bio.getvalue()
    req = urllib.request.Request(
        url2,
        data=data,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read(256)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        if raise_on_error:
            raise RuntimeError(f"discord webhook HTTPError {e.code}: {msg}") from e
        print(f"[discord] HTTPError {e.code}: {msg}")
    except Exception as e:
        if raise_on_error:
            raise
        print(f"[discord] error posting webhook: {e}")


def server_embed(info: Dict[str, Any]) -> Dict[str, Any]:
    server = (info or {}).get("server", {}) or {}
    cfg = (info or {}).get("config", {}) or {}
    plugins = (info or {}).get("plugins", {}) or {}

    name = str(server.get("name") or "FLEXMD")
    location = str(server.get("location") or "")
    host = str(cfg.get("host") or "")
    port = str(cfg.get("port") or "")

    avail = []
    for k, v in plugins.items():
        ok = bool((v or {}).get("available"))
        avail.append((str(k), ok))
    avail.sort(key=lambda kv: kv[0])

    ok_count = sum(1 for _, ok in avail if ok)
    total = len(avail)

    desc = []
    if location:
        desc.append(f"Location: `{location}`")
    if host and port:
        desc.append(f"Listening: `{host}:{port}`")
    if total:
        desc.append(f"Plugins available: `{ok_count}/{total}`")

    fields: List[Dict[str, Any]] = []
    if total:
        lines = [f"{'✅' if ok else '❌'} `{k}`" for k, ok in avail]
        fields.append({"name": "Backends", "value": "\n".join(lines)[:1000], "inline": False})

    return {
        "title": f"{name} started",
        "description": "\n".join(desc)[:3900],
        "fields": fields[:25],
        "timestamp": _iso_utc_now(),
    }

def simulate_embed(
    *,
    backend: str,
    selected_backend: str,
    n_atoms: int,
    n_steps: int,
    dt_ps: float,
    report_stride: int,
    cache_hit: bool,
    result_id: str = "",
    duration_s: float = 0.0,
) -> Dict[str, Any]:
    lines = [
        f"backend: `{backend}` → `{selected_backend}`",
        f"atoms: `{int(n_atoms)}`  steps: `{int(n_steps)}`  dt: `{dt_ps}` ps  stride: `{int(report_stride)}`",
        f"cache: `{'HIT' if cache_hit else 'MISS'}`",
    ]
    if result_id:
        lines.append(f"result_id: `{str(result_id)[:80]}`")
    if duration_s and duration_s > 0:
        lines.append(f"runtime: `{duration_s:.3f}s`")

    return {
        "title": "FLEXMD simulate",
        "description": "\n".join(lines)[:3900],
        "timestamp": _iso_utc_now(),
    }

def _get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default

def _fmt_float(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _titlecase_simple(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.islower():
        return s[:1].upper() + s[1:]
    return s

def kit_embed(manifest: Dict[str, Any], *, title: Optional[str] = None) -> Dict[str, Any]:
    m = manifest or {}
    ident = _get(m, "identity", {}) or {}
    scaling = _get(m, "scaling", {}) or {}
    voxel = _get(m, "voxel", {}) or {}
    splitting = _get(m, "splitting", {}) or {}
    geometry = _get(m, "geometry", {}) or {}
    stats = _get(m, "stats", {}) or {}
    files = _get(m, "files", {}) or {}
    pkg = _get(m, "package", {}) or {}

    fallback = None
    if isinstance(ident, dict):
        fallback = _get(ident, "name") or _get(ident, "formula")
    mol = title or fallback or "Molecule"
    mol = _titlecase_simple(str(mol))
    embed_title = f"{mol} 3D Print Kit"

    schema = _get(m, "schema", "flexmd.spacefill_kit")
    attachment = _get(pkg, "filename", None) or _get(files, "full_reference", None) or "spacefill_kit.zip"
    surface = _get(geometry, "surface_method", "unknown")
    seams = _get(geometry, "seams", "")

    desc_lines = [
        f"Schema: `{schema}`",
        f"Attachment: `{attachment}`",
        f"Surface: `{surface}`",
    ]
    if seams:
        desc_lines.append(f"Seams: `{seams}`")
    description = "\n".join(desc_lines)[:3900]

    atoms_n = stats.get("atoms")
    parts_list = files.get("parts") if isinstance(files.get("parts"), list) else []
    parts_count = len(parts_list)

    split_mode = _get(splitting, "split_mode", "unknown")
    target_mm = _get(scaling, "target_max_mm", None)
    scale_mmA = _get(scaling, "scale_mm_per_A", None)
    vox_mm = _get(voxel, "voxel_size_mm_used", None)
    gap_mm = _get(splitting, "gap_mm", None)

    bbox = _get(geometry, "bbox_mm", {}) or {}
    bbox_dim = _get(bbox, "max_dim", None)

    fields: List[Dict[str, Any]] = []

    def _f(name: str, value: str, inline: bool = True) -> None:
        fields.append({"name": name[:256], "value": value[:1024], "inline": inline})

    _f("Atoms", f"`{atoms_n}`" if atoms_n is not None else "`?`", True)
    _f("Parts", f"`{parts_count}`", True)
    _f("Split mode", f"`{split_mode}`", True)

    if target_mm is not None:
        _f("Target size", f"`{_fmt_float(target_mm, 3)}` mm", True)
    else:
        _f("Target size", "`?`", True)

    if scale_mmA is not None:
        _f("Scale", f"`{_fmt_float(scale_mmA, 3)}` mm/Å", True)
    else:
        _f("Scale", "`?`", True)

    if vox_mm is not None:
        _f("Voxel", f"`{_fmt_float(vox_mm, 3)}` mm", True)
    else:
        _f("Voxel", "`?`", True)

    if gap_mm is not None:
        _f("Gap", f"`{_fmt_float(gap_mm, 3)}` mm", True)
    else:
        _f("Gap", "`?`", True)

    if bbox_dim is not None:
        _f("BBox", f"`{_fmt_float(bbox_dim, 2)}` mm", True)
    else:
        _f("BBox", "`?`", True)

    if isinstance(ident, dict) and ident:
        id_bits = []
        for k in ("formula", "name", "smiles", "inchikey"):
            v = ident.get(k)
            if isinstance(v, str) and v.strip():
                id_bits.append(f"{k}: `{v.strip()[:180]}`")
        if id_bits:
            _f("Identity", "\n".join(id_bits)[:1024], inline=False)

    if parts_list:
        lines: List[str] = []
        for p in parts_list:
            if not isinstance(p, dict):
                continue
            fn = p.get("filename") or "part.stl"
            lbl = p.get("label") or ""
            base = "🧩" if p.get("is_base") else "•"
            col = p.get("recommended_color") if isinstance(p.get("recommended_color"), dict) else {}
            cname = (col.get("name") or "").strip()
            chex = (col.get("hex") or "").strip()
            color_txt = " ".join(x for x in [cname, chex] if x).strip()

            if color_txt:
                lines.append(f"{base} `{fn}` — `{lbl}` — **{color_txt}**")
            else:
                lines.append(f"{base} `{fn}` — `{lbl}`")

        out = ""
        shown = 0
        for ln in lines:
            if len(out) + len(ln) + 1 > 950:
                break
            out += (ln + "\n")
            shown += 1
        out = out.strip()
        if shown < len(lines):
            out += f"\n… +{len(lines) - shown} more"
        _f("Parts list", out[:1024], inline=False)

    return {
        "title": embed_title[:256],
        "description": description,
        "fields": fields[:25],
        "timestamp": _iso_utc_now(),
        "footer": {"text": "FLEXMD • space-fill kit"},
    }


def kit_embeds(manifest: Dict[str, Any], *, title: Optional[str] = None) -> List[Dict[str, Any]]:
    return [kit_embed(manifest, title=title)]
