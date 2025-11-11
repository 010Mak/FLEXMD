from __future__ import annotations
import json, urllib.request, urllib.error, urllib.parse
from typing import Any, Dict, List, Optional

def _valid_discord_webhook(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url or "")
        parts = [x for x in p.path.split("/") if x]
        return (
            p.scheme in ("http", "https")
            and p.netloc.endswith("discord.com")
            and len(parts) >= 4
            and parts[0] == "api"
            and parts[1] == "webhooks"
        )
    except Exception:
        return False

def post(
    url: str,
    *,
    content: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
    wait: bool = True,
    username: Optional[str] = None
) -> None:
    if not _valid_discord_webhook(url):
        print("[discord] invalid webhook URL; skipping")
        return

    if wait:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}wait=true"

    payload: Dict[str, Any] = {}
    if content:
        payload["content"] = content[:1900]
    if embeds:
        payload["embeds"] = embeds[:10]
    if username:
        payload["username"] = username

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.getcode()
            body = resp.read(120).decode("utf-8", "ignore")
            print(f"[discord] posted ({code}) {body[:120]}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        print(f"[discord] HTTPError {e.code}: {msg}")
    except Exception as e:
        print(f"[discord] error posting webhook: {e}")
