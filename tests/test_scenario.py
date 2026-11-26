import dataclasses

import numpy as np
import pytest

from whnav.env.scenario import Scenario, make_scenario
from whnav.env.workers import Heading, WorkerState

CORRIDOR = ["#######", "#.....#", "#######"]
WAREHOUSE = [
    "###########",
    "#.........#",
    "#.##.##.#.#",
    "#.........#",
    "#.##.##.#.#",
    "#.........#",
    "###########",
]


def warehouse_scenario(pattern="consecutive", seed=0):
    return make_scenario(
        WAREHOUSE,
        (1, 1),
        (5, 9),
        [WorkerState((3, 5), Heading.E), WorkerState((5, 2), Heading.N)],
        p=0.7,
        worker_seed=42,
        pattern=pattern,
        d=2 if pattern == "consecutive" else 0,
        f=0.25 if pattern != "none" else 0.0,
        schedule_rng=np.random.default_rng(seed),
        scenario_id=f"dev-test-{pattern}",
    )


def test_make_scenario_computes_path_and_horizon():
    sc = make_scenario(CORRIDOR, (1, 1), (1, 5))
    assert (sc.shortest_path, sc.horizon) == (4, 8)
    assert sc.schedule == (False,) * 8 and sc.f_realized == 0.0


@pytest.mark.parametrize("pattern", ["none", "scattered", "consecutive"])
def test_json_round_trip_is_lossless(pattern):
    sc = warehouse_scenario(pattern)
    back = Scenario.from_json(sc.to_json())
    assert back == sc
    assert hash(back) == hash(sc)
    assert back.grid.blocked.tolist() == sc.grid.blocked.tolist()
    assert "\n" not in sc.to_json()


def test_scattered_scenario_has_d_1_and_isolated_drops():
    sc = warehouse_scenario("scattered")
    assert sc.d == 1
    s = np.array(sc.schedule)
    assert s.any() and not (s[1:] & s[:-1]).any()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split": "training"}, "split"),
        ({"pattern": "bursty"}, "pattern"),
        ({"p": 1.5}, "p must"),
        ({"start": (0, 0)}, "start"),
        ({"goal": (1, 1)}, "differ"),
        ({"shortest_path": 5}, "shortest_path"),
        ({"horizon": 9}, "horizon"),
        ({"schedule": (False,) * 7}, "length"),
        ({"workers": (WorkerState((0, 3), Heading.E),)}, "worker"),
        ({"d": 1}, "pattern 'none'"),
        ({"schedule": (True,) + (False,) * 7}, "pattern 'none'"),
    ],
)
def test_invalid_scenarios_are_rejected(changes, message):
    base = make_scenario(CORRIDOR, (1, 1), (1, 5))
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **changes)


def test_unreachable_goal_is_rejected():
    with pytest.raises(ValueError, match="reachable"):
        make_scenario(["#####", "#.#.#", "#####"], (1, 1), (1, 3))


def test_outage_blocks_must_have_length_d():
    good = make_scenario(
        CORRIDOR,
        (1, 1),
        (1, 5),
        pattern="consecutive",
        d=2,
        f=0.25,
        schedule=[False, True, True, False, False, False, False, False],
    )
    assert good.f_realized == 0.25
    with pytest.raises(ValueError, match="exactly d"):
        make_scenario(
            CORRIDOR,
            (1, 1),
            (1, 5),
            pattern="consecutive",
            d=2,
            f=0.25,
            schedule=[False, True, True, True, False, False, False, False],
        )


def test_outage_pattern_needs_a_schedule_source():
    with pytest.raises(ValueError, match="schedule"):
        make_scenario(CORRIDOR, (1, 1), (1, 5), pattern="scattered", f=0.25)
