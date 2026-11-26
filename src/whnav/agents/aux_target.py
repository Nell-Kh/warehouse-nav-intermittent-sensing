"""Shared interface for the auxiliary occupancy target (SPEC §4)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from whnav.env.grid import Cell, GridMap

AUX_SIZE = 9


def aux_target(
    grid: GridMap,
    robot: Cell,
    worker_positions: Sequence[Cell],
    outage: bool,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return occupancy labels and a validity mask for the robot-centered 9×9 area.

    Coordinates follow the map's (row, column) convention, with the robot at
    index (4, 4). The boolean mask covers traversable, in-map cells outside the
    current 7×7 sensor window. During an outage it covers every traversable,
    in-map cell in the area. Earlier readings do not affect the mask.

    The float32 labels are 1.0 at masked cells with one or more workers and 0.0
    elsewhere. Worker positions are privileged ground truth for supervision and
    prediction-quality evaluation; they are never supplied to the policy.
    """
    raise NotImplementedError
