"""Talk to the running Blender over its own socket.

    python scripts/blender_send.py code some_script.py
    python scripts/blender_send.py shot data/blender.png
"""
import json
import socket
import sys
from pathlib import Path

HOST, PORT = "127.0.0.1", 9876


def send(command: str, params: dict, timeout: float = 120.0):
    sock = socket.create_connection((HOST, PORT), timeout=10)
    sock.settimeout(timeout)
    sock.sendall(json.dumps({"type": command, "params": params}).encode())
    chunks = b""
    while True:
        chunk = sock.recv(1 << 16)
        if not chunk:
            break
        chunks += chunk
        try:
            return json.loads(chunks.decode())
        except json.JSONDecodeError:
            continue
    sock.close()
    return json.loads(chunks.decode())


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "code":
        reply = send("execute_code", {"code": Path(sys.argv[2]).read_text(encoding="utf-8")})
    elif what == "shot":
        reply = send("get_viewport_screenshot",
                     {"filepath": str(Path(sys.argv[2]).resolve()), "max_size": 1200})
    else:
        reply = send(what, {})
    print(json.dumps(reply)[:4000])
