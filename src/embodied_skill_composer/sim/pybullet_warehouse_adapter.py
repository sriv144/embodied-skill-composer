from __future__ import annotations

from typing import Any

from embodied_skill_composer.sim.mock_warehouse_adapter import MockWarehouseAdapter


class PyBulletWarehouseAdapter(MockWarehouseAdapter):
    """
    Minimal PyBullet-backed warehouse adapter.

    The collection logic intentionally reuses the deterministic warehouse state machine so the
    planner, perception, and benchmark pipeline stay the same across adapters. When `pybullet`
    is installed, this adapter can be extended with richer rendering and collision checking
    without changing the public interfaces.
    """

    def __init__(
        self, runtime_config: dict[str, Any], scene_config: dict[str, Any], gui: bool = False
    ) -> None:
        self.gui = gui
        self._pybullet = None
        try:
            import pybullet as p
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "PyBullet warehouse adapter requested, but pybullet is not installed. "
                "Use the mock warehouse backend or install the optional pybullet environment."
            ) from exc
        self._pybullet = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        super().__init__(runtime_config=runtime_config, scene_config=scene_config)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        assert self._pybullet is not None
        p = self._pybullet
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

    def close(self) -> None:
        if self._pybullet is not None and self._pybullet.isConnected(self.client):
            self._pybullet.disconnect(self.client)
