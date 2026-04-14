from pathlib import Path

from embodied_skill_composer.core.executor import TaskExecutor
from embodied_skill_composer.core.planner import RuleBasedPlanner
from embodied_skill_composer.sim.mock_adapter import MockTabletopAdapter
from embodied_skill_composer.tasks.catalog import load_tasks


def test_pick_and_place_integration(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    import yaml

    runtime_config = yaml.safe_load((project_root / "configs" / "runtime.yaml").read_text())
    scene_config = yaml.safe_load((project_root / "configs" / "scene.yaml").read_text())
    tasks = load_tasks(project_root / "configs" / "tasks.yaml")
    adapter = MockTabletopAdapter(runtime_config=runtime_config, scene_config=scene_config)
    try:
        task = tasks["pick_and_place_red_to_tray"]
        planner = RuleBasedPlanner()
        plan = planner.plan(task, adapter.get_world_state())
        executor = TaskExecutor(adapter=adapter, log_dir=tmp_path)
        report = executor.run(task, plan)

        assert report.success is True
        assert report.log_path is not None
        final_pos = report.final_world.objects["red_block"].position
        tray_center = report.final_world.zones["tray"].center
        assert abs(final_pos[0] - tray_center[0]) < 0.05
        assert abs(final_pos[1] - tray_center[1]) < 0.05
    finally:
        adapter.close()
