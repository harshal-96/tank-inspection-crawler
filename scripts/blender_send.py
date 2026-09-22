"""Send a Python file to a running Blender for execution, and print what it says.

The bundled Blender MCP tool times out against this addon, but the addon's own
socket answers fine: it speaks a small JSON protocol on port 9876. Anything the
executed code prints comes back, so a builder script reports by printing JSON.

    python scripts/blender_send.py tank/sim/blender_tank.py
    python scripts/blender_send.py --code "import bpy; print(len(bpy.data.objects))"
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

HOST = "127.0.0.1"
PORT = 9876


def talk(payload: dict, timeout: float = 180.0) -> dict:
    """One request, one reply. The addon answers with a single JSON object."""
    with socket.create_connection((HOST, PORT), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(json.dumps(payload).encode())
        chunks: list[bytes] = []
        while True:
            block = sock.recv(1 << 16)
            if not block:
                break
            chunks.append(block)
            try:
                return json.loads(b"".join(chunks).decode())
            except json.JSONDecodeError:
                continue          # a reply larger than one recv
    raise RuntimeError("Blender closed the connection without a complete reply")


def run_code(code: str, timeout: float = 180.0) -> dict:
    return talk({"type": "execute_code", "params": {"code": code}}, timeout=timeout)


def run_file(path: Path, timeout: float = 180.0) -> dict:
    """Run a file by path inside Blender, so the payload stays small.

    Passing the source over the socket works until the script grows; running it
    by path also means Blender reports the real filename in any traceback.
    """
    target = path.resolve().as_posix()
    code = (
        "import traceback\n"
        f"path = r'{target}'\n"
        "try:\n"
        "    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), "
        "{'__name__': '__main__', '__file__': path})\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
    )
    return run_code(code, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", help="Python file to run inside Blender")
    parser.add_argument("--code", help="Run this source instead of a file")
    parser.add_argument("--scene", action="store_true", help="Just report the scene")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.scene:
        reply = talk({"type": "get_scene_info", "params": {}}, timeout=args.timeout)
    elif args.code:
        reply = run_code(args.code, timeout=args.timeout)
    elif args.script:
        reply = run_file(Path(args.script), timeout=args.timeout)
    else:
        parser.error("give a script, --code, or --scene")

    if reply.get("status") != "success":
        print(json.dumps(reply, indent=1)[:4000])
        return 1
    out = reply.get("result")
    if isinstance(out, dict):
        printed = out.get("result", "")
        print(printed if printed else json.dumps(out, indent=1)[:4000])
    else:
        print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
