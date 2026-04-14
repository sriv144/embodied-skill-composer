# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.gpu import inspect_gpu_runtime
from embodied_skill_composer.assembly.runtime import load_runtime_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate torch and CUDA visibility for an assembly runtime profile.")
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    status = inspect_gpu_runtime(runtime_profile)
    print(f"Runtime profile: {status.runtime_profile} ({status.backend})")
    print(f"Requested device: {status.requested_device or 'auto'}")
    print(f"Torch installed: {status.torch_installed}")
    print(f"CUDA available: {status.cuda_available}")
    print(f"Selected device: {status.selected_device}")
    if status.device_name:
        print(f"Device name: {status.device_name}")
    print(f"Tensor allocation OK: {status.tensor_allocation_ok}")
    print(json.dumps(status.model_dump(mode='json'), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
