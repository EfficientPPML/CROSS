"""Suite stage 7: hand the resident service the uploaded ciphertext bundle."""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib

TIMEOUT = float(os.environ.get("P3_REQUEST_TIMEOUT", "14400"))


def request(paths, payload):
    last = None
    for attempt in range(10):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.settimeout(TIMEOUT)
            conn.connect(str(paths.socket_path))
            break
        except OSError as error:
            last = error
            conn.close()
            time.sleep(0.5 * (attempt + 1))
    else:
        sys.exit(f"[p3] cannot reach the service at {paths.socket_path}: {last}")
    try:
        conn.sendall((json.dumps(payload) + "\n").encode())
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                if not chunks:
                    sys.exit("[p3] service closed the connection without replying")
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    except socket.timeout:
        sys.exit(f"[p3] no reply within {TIMEOUT:.0f}s")
    finally:
        conn.close()
    return json.loads(b"".join(chunks).decode())


def main():
    size = int(sys.argv[1])
    paths = lib.Paths(size)
    if not paths.socket_path.exists():
        sys.exit("[p3] no resident service; stage 3 must run first")
    started = time.perf_counter()
    response = request(paths, {"cmd": "infer", "indir": str(paths.upload),
                               "outdir": str(paths.download)})
    if not response.get("ok"):
        sys.exit(f"[p3] encrypted compute failed: {response.get('error')}")
    r = response["result"]
    stage_s = time.perf_counter() - started
    print(f"         [p3] {r['batch']} image(s) on {r['devices']} chip(s): "
          f"{r['execute_s']:.4f}s accelerator time -> "
          f"{r['per_inference_s'] * 1000:.3f} ms/image, "
          f"{r['images_per_s']:.2f} images/s")
    lib.report_steps(paths, {
        "server_encrypted_compute_s": stage_s,
        "server_execute_s": r["execute_s"],
        "server_per_inference_s": r["per_inference_s"],
        "server_images_per_s": r["images_per_s"],
        "server_bundle_load_s": r["load_s"],
        "server_bundle_store_s": r["store_s"],
        "server_result_ciphertexts": r["result_ciphertexts"]})


if __name__ == "__main__":
    main()
