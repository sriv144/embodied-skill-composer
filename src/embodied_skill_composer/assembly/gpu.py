from __future__ import annotations

import importlib.util

from embodied_skill_composer.assembly.models import AssemblyRuntimeProfile, GpuRuntimeStatus


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def inspect_gpu_runtime(runtime_profile: AssemblyRuntimeProfile) -> GpuRuntimeStatus:
    requested_device = runtime_profile.device
    if not torch_available():
        return GpuRuntimeStatus(
            runtime_profile=runtime_profile.name,
            backend=runtime_profile.backend,
            requested_device=requested_device,
            torch_installed=False,
            cuda_available=False,
            selected_device="cpu",
            tensor_allocation_ok=False,
            notes=[
                "Torch is not installed in the active environment.",
                "Install `requirements-rl.txt` before using learned-policy training or CUDA validation.",
            ],
        )

    import torch

    cuda_available = torch.cuda.is_available()
    selected_device = requested_device or ("cuda" if cuda_available else "cpu")
    device_name = None
    tensor_allocation_ok = False
    notes: list[str] = [runtime_profile.notes] if runtime_profile.notes else []

    if selected_device.startswith("cuda"):
        if not cuda_available:
            notes.append("CUDA is not available to torch in the active environment.")
        else:
            device_index = 0
            if ":" in selected_device:
                _, _, suffix = selected_device.partition(":")
                if suffix.isdigit():
                    device_index = int(suffix)
            device_name = torch.cuda.get_device_name(device_index)
            try:
                tensor = torch.ones((4, 4), device=selected_device)
                tensor_allocation_ok = bool(float(tensor.sum().item()) == 16.0)
                notes.append("CUDA tensor allocation succeeded.")
            except Exception as exc:  # pragma: no cover - hardware-specific path
                notes.append(f"CUDA tensor allocation failed: {exc}")
    else:
        tensor_allocation_ok = True
        notes.append("Runtime profile is configured for CPU execution.")
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            notes.append("CUDA is available and can be used by switching to a GPU runtime profile.")

    if runtime_profile.requires_linux:
        notes.append("This runtime profile expects a Linux host.")
    if runtime_profile.requires_nvidia_gpu:
        notes.append("This runtime profile expects a CUDA-capable NVIDIA GPU.")

    return GpuRuntimeStatus(
        runtime_profile=runtime_profile.name,
        backend=runtime_profile.backend,
        requested_device=requested_device,
        torch_installed=True,
        cuda_available=cuda_available,
        selected_device=selected_device,
        device_name=device_name,
        tensor_allocation_ok=tensor_allocation_ok,
        notes=notes,
    )
