from statistics import mean

import numpy as np

from embodied_skill_composer.construction.workbench_evidence import _cli_validation_mean


def test_cli_validation_mean_matches_frozen_large_makespan_summation() -> None:
    values = [
        8032.021841623069,
        9543.530203540338,
        8330.082462434548,
        9991.606343051048,
        9794.982533825698,
        10821.713373701316,
        9191.153948801979,
        10485.058762248798,
        10243.60937206579,
        8359.390999293191,
    ]
    frozen_mean = float(np.mean(values))
    assert abs(frozen_mean - mean(values)) > 1e-12
    assert _cli_validation_mean(values) == frozen_mean
