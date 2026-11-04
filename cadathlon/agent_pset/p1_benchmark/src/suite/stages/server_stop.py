"""Release the accelerator held by the resident service (not a harness stage)."""
import os
import socket
import sys

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else lib.read_active()["size"]
    paths = lib.Paths(size)
    if not paths.socket_path.exists():
        print("[p3] no resident service")
        return
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(120)
    conn.connect(str(paths.socket_path))
    conn.sendall(b'{"cmd": "stop"}\n')
    conn.recv(4096)
    conn.close()
    print("[p3] service stopped")


if __name__ == "__main__":
    main()
