"""Thin client for the xaitalk remote /execute endpoint (Colab-via-ngrok).

Used to drive YOLO training on a remote A100 since the ngrok tunnel only
exposes a single POST /execute. No file-upload endpoint, so transfers go
through chunked base64 inside execute calls.

Examples:
    python tools/remote_exec.py exec 'import torch; print(torch.cuda.is_available())'
    python tools/remote_exec.py upload local.zip /content/remote.zip
    python tools/remote_exec.py download /content/best.pt local_best.pt
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

DEFAULT_URL = "https://areological-immunogenically-milania.ngrok-free.dev"
CHUNK = 4 * 1024 * 1024  # 4 MiB raw per chunk -> ~5.4 MiB base64 in JSON


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/execute",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def execute(code: str, *, url: str = DEFAULT_URL, timeout: int = 600,
            print_output: bool = True) -> dict:
    """Run code on remote, return the full result dict. Optionally print stdout/stderr."""
    res = _post(url, {"code": code}, timeout=timeout)
    if print_output:
        if res.get("stdout"):
            sys.stdout.write(res["stdout"])
            sys.stdout.flush()
        if res.get("stderr"):
            sys.stderr.write(res["stderr"])
            sys.stderr.flush()
        if not res.get("success"):
            sys.stderr.write(f"\n[remote error] {res.get('error')}\n{res.get('traceback') or ''}\n")
    return res


def upload(local: Path, remote: str, *, url: str = DEFAULT_URL) -> None:
    """Chunk-upload a local file to a remote path. Verifies sha256 at the end."""
    local = Path(local)
    size = local.stat().st_size
    sha = hashlib.sha256()

    # init: open the remote file in write-binary mode and remember the handle
    init = f"""
import os
os.makedirs(os.path.dirname({remote!r}) or '.', exist_ok=True)
_upload_fh = open({remote!r}, 'wb')
print('opened')
"""
    res = execute(init, url=url, print_output=False)
    if not res.get("success"):
        raise RuntimeError(f"upload init failed: {res.get('error')}\n{res.get('traceback')}")

    sent = 0
    t0 = time.time()
    with local.open("rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            sha.update(buf)
            b64 = base64.b64encode(buf).decode("ascii")
            code = f"_upload_fh.write(__import__('base64').b64decode({b64!r}))"
            res = execute(code, url=url, print_output=False, timeout=120)
            if not res.get("success"):
                raise RuntimeError(f"chunk failed at offset {sent}: {res.get('error')}")
            sent += len(buf)
            mb = sent / (1024 * 1024)
            elapsed = time.time() - t0
            rate = mb / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  {mb:7.1f} / {size/(1024*1024):7.1f} MiB  ({rate:5.2f} MiB/s)")
            sys.stdout.flush()
    print()

    finalize = f"""
_upload_fh.close()
import hashlib, os
_h = hashlib.sha256()
with open({remote!r}, 'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        _h.update(chunk)
print('size', os.path.getsize({remote!r}))
print('sha256', _h.hexdigest())
"""
    res = execute(finalize, url=url, print_output=False)
    out = res.get("stdout", "")
    if not res.get("success"):
        raise RuntimeError(f"upload finalize failed: {out}")
    remote_sha = next((l.split()[1] for l in out.splitlines() if l.startswith("sha256")), None)
    if remote_sha != sha.hexdigest():
        raise RuntimeError(f"sha mismatch: local={sha.hexdigest()} remote={remote_sha}")
    print(f"  ok ({size/(1024*1024):.1f} MiB, sha256 {remote_sha[:12]}...)")


def download(remote: str, local: Path, *, url: str = DEFAULT_URL) -> None:
    """Chunk-download a remote file to a local path, sha-verified."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)

    info = execute(f"""
import os, hashlib
_size = os.path.getsize({remote!r})
_h = hashlib.sha256()
with open({remote!r}, 'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        _h.update(chunk)
print('size', _size)
print('sha256', _h.hexdigest())
""", url=url, print_output=False)
    if not info.get("success"):
        raise RuntimeError(info.get("traceback") or info.get("error"))
    out = info.get("stdout", "")
    size = int(next(l.split()[1] for l in out.splitlines() if l.startswith("size")))
    remote_sha = next(l.split()[1] for l in out.splitlines() if l.startswith("sha256"))

    sha = hashlib.sha256()
    t0 = time.time()
    pos = 0
    with local.open("wb") as f:
        while pos < size:
            n = min(CHUNK, size - pos)
            code = f"""
import base64
with open({remote!r}, 'rb') as _f:
    _f.seek({pos})
    print(base64.b64encode(_f.read({n})).decode('ascii'))
"""
            res = execute(code, url=url, print_output=False, timeout=120)
            if not res.get("success"):
                raise RuntimeError(f"download chunk at {pos} failed")
            chunk_b64 = res["stdout"].strip()
            buf = base64.b64decode(chunk_b64)
            f.write(buf)
            sha.update(buf)
            pos += len(buf)
            mb = pos / (1024 * 1024)
            elapsed = time.time() - t0
            rate = mb / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  {mb:7.1f} / {size/(1024*1024):7.1f} MiB  ({rate:5.2f} MiB/s)")
            sys.stdout.flush()
    print()

    if sha.hexdigest() != remote_sha:
        raise RuntimeError(f"sha mismatch: local={sha.hexdigest()} remote={remote_sha}")
    print(f"  ok ({size/(1024*1024):.1f} MiB, sha256 {remote_sha[:12]}...)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exec"); e.add_argument("code")
    u = sub.add_parser("upload"); u.add_argument("local"); u.add_argument("remote")
    d = sub.add_parser("download"); d.add_argument("remote"); d.add_argument("local")
    args = ap.parse_args()

    if args.cmd == "exec":
        execute(args.code, url=args.url)
    elif args.cmd == "upload":
        upload(Path(args.local), args.remote, url=args.url)
    elif args.cmd == "download":
        download(args.remote, Path(args.local), url=args.url)


if __name__ == "__main__":
    main()
