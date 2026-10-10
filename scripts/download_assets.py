#!/usr/bin/env python3
"""Download test assets for robine-nam into ./assets (stdlib only).

Sources:
  * DI inputs and IRs : public demo files from TONE3000's neural-amp-modeler-wasm
                        repository on GitHub (no authentication).
  * Upstream examples : example .nam models from NeuralAmpModelerCore (MIT).
  * TONE3000 models   : A2 (.nam) models of given tones through the TONE3000 API.
                        Needs a TONE3000 *publishable* key (t3k_pub_...) in
                        TONE3000_CLIENT_ID; the script runs the OAuth device flow
                        (you approve once in a browser) and caches the token in
                        ~/.config/robine-nam/tone3000-token.json. A ready-made
                        TONE3000_ACCESS_TOKEN is used directly when set.

Environment variables are read from the process environment, then from a
`.env` file at the repository root (or --env-file); see .env.example.
Variables already set in the environment win over the file.

Examples:
  scripts/download_assets.py                       # inputs subset + IRs + upstream examples
  scripts/download_assets.py --all-inputs
  scripts/download_assets.py --tone 6580 --tone https://www.tone3000.com/tones/dumble-steel-string-singer-29285
  scripts/download_assets.py --tone 6580 --search "AC30"  # search then pick ids

Downloaded TONE3000 models stay local: most are licensed for personal use
only (T3K license) and must not be committed or redistributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

GITHUB_API = "https://api.github.com/repos/tone-3000/neural-amp-modeler-wasm/contents"
GITHUB_REF = "main"
UPSTREAM_RAW = "https://raw.githubusercontent.com/sdatkinson/NeuralAmpModelerCore"
UPSTREAM_COMMIT = "0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842"
UPSTREAM_EXAMPLES = ["A2.nam", "slimmable_container.nam", "wavenet_a2_feature_test.nam", "lstm.nam"]
TONE3000_API = "https://www.tone3000.com/api/v1"
TOKEN_CACHE = Path.home() / ".config" / "robine-nam" / "tone3000-token.json"
USER_AGENT = "robine-nam-assets/0.1 (+https://github.com/bailleul-dev/robine-nam)"

# A small, varied subset of the 35 public DI files (about 25 MB).
DEFAULT_INPUTS = [
    "Brit - Guitar.wav",
    "Mayer - Guitar.wav",
    "Pluck - Guitar.wav",
    "Metalcore - Guitar.wav",
    "Harmonics - Guitar.wav",
    "Downtown - Bass.wav",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments, no quoting, no override."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def http_json(url: str, headers: dict[str, str] | None = None, data: dict | None = None, timeout: float = 60.0):
    body = None
    hdrs = {"Accept": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    request = urllib.request.Request(url, headers=hdrs, data=body, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), response.status
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(error)}
        return payload, error.code


def download_file(url: str, destination: Path, headers: dict[str, str] | None = None, timeout: float = 120.0) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(destination)
    return destination.stat().st_size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Public GitHub assets
# ---------------------------------------------------------------------------

def download_github_dir(subdir: str, output: Path, wanted: list[str] | None, overwrite: bool, manifest: dict) -> None:
    listing, status = http_json(f"{GITHUB_API}/{subdir}?ref={GITHUB_REF}", headers={"Accept": "application/vnd.github+json"})
    if status != 200 or not isinstance(listing, list):
        raise SystemExit(f"GitHub listing of {subdir} failed: HTTP {status} {listing}")
    files = [item for item in listing if item.get("type") == "file" and item["name"].lower().endswith(".wav")]
    if wanted is not None:
        files = [item for item in files if item["name"] in wanted]
        missing = set(wanted) - {item["name"] for item in files}
        if missing:
            log(f"warning: not found upstream: {sorted(missing)}")
    entries = []
    for item in sorted(files, key=lambda f: f["name"].lower()):
        destination = output / item["name"]
        if destination.exists() and not overwrite:
            log(f"  keep  {destination}")
        else:
            log(f"  get   {item['name']} ({item['size']} bytes)")
            download_file(item["download_url"], destination)
        entries.append({
            "name": item["name"],
            "path": str(destination.relative_to(output.parent.parent)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_url": item.get("html_url"),
        })
    manifest[subdir.split("/")[-1]] = {
        "provider": "TONE3000 neural-amp-modeler-wasm demo files",
        "source": f"https://github.com/tone-3000/neural-amp-modeler-wasm/tree/{GITHUB_REF}/{subdir}",
        "license_note": "Public demo assets; for local testing only, do not redistribute.",
        "files": entries,
    }


def download_upstream_examples(output: Path, overwrite: bool, manifest: dict) -> None:
    entries = []
    for name in UPSTREAM_EXAMPLES:
        destination = output / name
        url = f"{UPSTREAM_RAW}/{UPSTREAM_COMMIT}/example_models/{name}"
        if destination.exists() and not overwrite:
            log(f"  keep  {destination}")
        else:
            log(f"  get   {name}")
            download_file(url, destination)
        entries.append({
            "name": name,
            "path": str(destination.relative_to(output.parent.parent)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_url": url,
        })
    manifest["upstream_examples"] = {
        "provider": "NeuralAmpModelerCore example_models",
        "commit": UPSTREAM_COMMIT,
        "license": "MIT",
        "files": entries,
    }


# ---------------------------------------------------------------------------
# TONE3000
# ---------------------------------------------------------------------------

def load_cached_token() -> dict | None:
    try:
        return json.loads(TOKEN_CACHE.read_text())
    except (OSError, ValueError):
        return None


def save_token(token: dict) -> None:
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    token = dict(token)
    token["obtained_at"] = time.time()
    TOKEN_CACHE.write_text(json.dumps(token, indent=2))
    try:
        TOKEN_CACHE.chmod(0o600)
    except OSError:
        pass


def refresh_token(client_id: str, cached: dict) -> dict | None:
    refresh = cached.get("refresh_token")
    if not refresh:
        return None
    payload, status = http_json(f"{TONE3000_API}/oauth/token", data={
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": client_id,
    })
    if status == 200 and "access_token" in payload:
        payload.setdefault("refresh_token", refresh)
        save_token(payload)
        return payload
    log(f"token refresh failed (HTTP {status}); re-authorizing")
    return None


def device_flow(client_id: str, open_browser: bool) -> dict:
    payload, status = http_json(f"{TONE3000_API}/oauth/device_authorization", data={"client_id": client_id})
    if status != 200 or "device_code" not in payload:
        raise SystemExit(f"device authorization failed: HTTP {status} {payload}")
    url = payload.get("verification_uri_complete") or payload["verification_uri"]
    log("")
    log("TONE3000 authorization required. Open this URL and approve the request:")
    log(f"    {url}")
    log(f"    code: {payload['user_code']}  (expires in {payload.get('expires_in', 600)} s)")
    log("")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    interval = float(payload.get("interval", 5))
    deadline = time.time() + float(payload.get("expires_in", 600))
    while time.time() < deadline:
        time.sleep(interval)
        token, status = http_json(f"{TONE3000_API}/oauth/token", data={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": payload["device_code"],
            "client_id": client_id,
        })
        if status == 200 and "access_token" in token:
            save_token(token)
            log("authorized; token cached in " + str(TOKEN_CACHE))
            return token
        error = token.get("error") if isinstance(token, dict) else None
        if error == "slow_down":
            interval += 5.0
        elif error in (None, "authorization_pending"):
            continue
        else:
            raise SystemExit(f"device flow failed: {token}")
    raise SystemExit("device flow timed out")


def get_access_token(args: argparse.Namespace) -> str:
    env_token = os.environ.get("TONE3000_ACCESS_TOKEN")
    if env_token:
        return env_token
    client_id = args.client_id or os.environ.get("TONE3000_CLIENT_ID")
    if not client_id:
        raise SystemExit("set TONE3000_CLIENT_ID (publishable key, t3k_pub_...) or TONE3000_ACCESS_TOKEN")
    cached = load_cached_token()
    if cached and not args.reauth:
        age = time.time() - float(cached.get("obtained_at", 0))
        expires_in = float(cached.get("expires_in", 3600))
        if age < expires_in - 60:
            return cached["access_token"]
        refreshed = refresh_token(client_id, cached)
        if refreshed:
            return refreshed["access_token"]
    return device_flow(client_id, open_browser=not args.no_browser)["access_token"]


def parse_tone_id(value: str) -> str:
    match = re.search(r"(\d+)\s*$", value.strip().rstrip("/"))
    if not match:
        raise SystemExit(f"cannot find a tone id in {value!r}")
    return match.group(1)


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "model"


def tone3000_search(token: str, query: str, page_size: int = 10) -> None:
    params = urllib.parse.urlencode({"query": query, "architecture": "2", "format": "nam", "page_size": page_size})
    payload, status = http_json(f"{TONE3000_API}/tones/search?{params}", headers={"Authorization": f"Bearer {token}"})
    if status != 200:
        raise SystemExit(f"search failed: HTTP {status} {payload}")
    for tone in payload.get("data", []):
        gear = tone.get("gear")
        if isinstance(gear, dict):
            gear = gear.get("name") or gear.get("slug")
        user = tone.get("user") or {}
        print(f"{tone.get('id'):>7}  {tone.get('title')!s:<50.50}  {str(gear):<10}  A2 models: {tone.get('a2_models_count')}  license: {tone.get('license')}  by {user.get('username') if isinstance(user, dict) else user}")


def download_tone(token: str, tone_id: str, output: Path, overwrite: bool, manifest: dict) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    tone, status = http_json(f"{TONE3000_API}/tones/{tone_id}?architecture=2", headers=headers)
    if status != 200:
        raise SystemExit(f"tone {tone_id}: HTTP {status} {tone}")
    title = tone.get("title") or f"tone-{tone_id}"
    tone_dir = output / f"{slug(title)}-{tone_id}"
    models = []
    page = 1
    while True:
        params = urllib.parse.urlencode({"tone_id": tone_id, "architecture": "2", "page": page, "page_size": 100})
        payload, status = http_json(f"{TONE3000_API}/models?{params}", headers=headers)
        if status != 200:
            raise SystemExit(f"tone {tone_id} models: HTTP {status} {payload}")
        models.extend(payload.get("data", []))
        if page >= int(payload.get("total_pages", 1) or 1):
            break
        page += 1
    if not models:
        log(f"tone {tone_id} ({title}): no A2 models")
        return
    log(f"tone {tone_id} ({title}): {len(models)} A2 model(s) -> {tone_dir}")
    entries = []
    for model in models:
        name = model.get("name") or f"model-{model.get('id')}"
        filename = slug(name)
        if not filename.lower().endswith(".nam"):
            filename += ".nam"
        destination = tone_dir / filename
        if destination.exists() and not overwrite:
            log(f"  keep  {destination.name}")
        else:
            log(f"  get   {destination.name}")
            download_file(model["model_url"], destination, headers=headers)
        entries.append({
            "model_id": model.get("id"),
            "name": name,
            "path": str(destination.relative_to(output.parent)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "architecture_version": model.get("architecture_version"),
            "size": model.get("size"),
        })
    user = tone.get("user") or {}
    manifest.setdefault("tone3000", []).append({
        "tone_id": tone_id,
        "title": title,
        "url": tone.get("url") or f"https://www.tone3000.com/tones/{tone_id}",
        "creator": user.get("username") if isinstance(user, dict) else user,
        "license": tone.get("license"),
        "gear": tone.get("gear"),
        "makes": tone.get("makes"),
        "tags": tone.get("tags"),
        "models": entries,
        "license_note": "Downloaded for local testing only. Do not commit or redistribute .nam files.",
    })


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--env-file", type=Path, default=repo_root / ".env", help="dotenv file (default: <repo>/.env)")
    parser.add_argument("--output", type=Path, default=None, help="asset directory (default: $ROBINE_NAM_ASSETS_DIR or <repo>/assets)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-inputs", action="store_true", help="skip the DI inputs")
    parser.add_argument("--all-inputs", action="store_true", help="download all 35 public DI files (~140 MB)")
    parser.add_argument("--no-irs", action="store_true")
    parser.add_argument("--no-upstream", action="store_true", help="skip the NeuralAmpModelerCore example models")
    parser.add_argument("--tone", action="append", default=[], metavar="ID_OR_URL", help="TONE3000 tone to download (repeatable)")
    parser.add_argument("--search", metavar="QUERY", help="search TONE3000 A2 tones and print ids")
    parser.add_argument("--client-id", help="TONE3000 publishable key (default: $TONE3000_CLIENT_ID)")
    parser.add_argument("--reauth", action="store_true", help="ignore the cached token")
    parser.add_argument("--no-browser", action="store_true", help="do not try to open the authorization URL")
    args = parser.parse_args()
    load_dotenv(args.env_file)

    output: Path = args.output or Path(os.environ.get("ROBINE_NAM_ASSETS_DIR") or repo_root / "assets")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            manifest = {}
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    if not args.no_inputs:
        log("== DI inputs")
        download_github_dir("ui/public/inputs", output / "inputs", None if args.all_inputs else DEFAULT_INPUTS, args.overwrite, manifest)
    if not args.no_irs:
        log("== impulse responses")
        download_github_dir("ui/public/irs", output / "irs", None, args.overwrite, manifest)
    if not args.no_upstream:
        log("== upstream example models")
        download_upstream_examples(output / "models" / "upstream", args.overwrite, manifest)

    if args.tone or args.search:
        token = get_access_token(args)
        if args.search:
            tone3000_search(token, args.search)
        if args.tone:
            manifest["tone3000"] = [t for t in manifest.get("tone3000", []) if str(t.get("tone_id")) not in {parse_tone_id(v) for v in args.tone}]
            for value in args.tone:
                download_tone(token, parse_tone_id(value), output / "models", args.overwrite, manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
