"""Suite stage; installed into the ml-inference checkout by install_suite.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib

import time


def main():
    size = int(sys.argv[1])
    paths = lib.Paths(size)
    devices = lib.device_count()
    lib.record_active(size, devices)
    started = time.perf_counter()
    packed = lib.pack_program()
    keys = lib.generate_keys(packed)
    lib.save_keys(keys, paths)
    ring = packed.ring_config
    print(f"         [p3] ring degree={ring.degree} slots={ring.num_slots} "
          f"depth={packed.depth} security={ring.target.bits}-bit "
          f"| devices={devices}")
    lib.report_steps(paths, {"client_key_generation_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
