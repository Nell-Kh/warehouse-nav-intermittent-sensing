import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from whnav.env.grid import Action
from whnav.env.observation import Channel, Scalar
from whnav.env.scenario import make_scenario
from whnav.env.warehouse_env import WarehouseEnv
from whnav.env.workers import Heading, WorkerState

N, S, E, W, WAIT = Action.N, Action.S, Action.E, Action.W, Action.WAIT
CORRIDOR = ["#######", "#.....#", "#######"]  # start (1,1), goal (1,5): L = 4, T = 8
WAREHOUSE = [
    "###########",
    "#.........#",
    "#.##.##.#.#",
    "#.........#",
    "#.##.##.#.#",
    "#.........#",
    "###########",
]


def corridor(workers=(), start=(1, 1), goal=(1, 5), **kw):
    return make_scenario(CORRIDOR, start, goal, list(workers), **kw)


def warehouse(seed=0, p=0.5, pattern="consecutive"):
    return make_scenario(
        WAREHOUSE,
        (1, 1),
        (5, 9),
        [
            WorkerState((3, 5), Heading.E),
            WorkerState((1, 7), Heading.W),
            WorkerState((5, 4), Heading.N),
        ],
        p=p,
        worker_seed=100 + seed,
        pattern=pattern,
        d=2,
        f=0.25,
        schedule_rng=np.random.default_rng(seed),
        scenario_id=f"dev-{seed}",
    )


def run(env, scenario, actions):
    """Run actions until the episode ends; return (rewards, infos, observations)."""
    obs, info = env.reset(options={"scenario": scenario})
    rewards, infos, observations = [], [info], [obs]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        assert trunc is False
        rewards.append(r)
        infos.append(info)
        observations.append(obs)
        if term:
            break
    return rewards, infos, observations


# --- Gymnasium compliance ---


def test_passes_gymnasium_env_checker():
    check_env(WarehouseEnv([warehouse(0), warehouse(1)]), skip_render_check=True)


def test_spaces():
    env = WarehouseEnv([warehouse()])
    assert env.observation_space["grid"].shape == (6, 7, 11)
    assert env.observation_space["scalars"].shape == (3,)
    assert env.action_space.n == 5


# --- episode outcomes and returns (SPEC §2.4, §2.5) ---


def test_empty_corridor_success_at_t_equals_l():
    rewards, infos, _ = run(WarehouseEnv([corridor()]), corridor(), [E] * 10)
    assert infos[-1]["outcome"] == "success" and infos[-1]["t"] == 4
    assert sum(rewards) == pytest.approx(1 - 0.5 * 4 / 8)


def test_always_wait_is_deadline_at_t_equals_t():
    rewards, infos, _ = run(WarehouseEnv([corridor()]), corridor(), [WAIT] * 20)
    assert infos[-1]["outcome"] == "deadline" and infos[-1]["t"] == 8
    assert sum(rewards) == pytest.approx(-0.5)


def test_goal_on_the_last_step_is_success():
    rewards, infos, _ = run(WarehouseEnv([corridor()]), corridor(), [WAIT] * 4 + [E] * 4)
    assert infos[-1]["outcome"] == "success" and infos[-1]["t"] == 8
    assert sum(rewards) == pytest.approx(0.5)


def test_same_cell_collision():
    sc = corridor([WorkerState((1, 3), Heading.W)])
    _, infos, _ = run(WarehouseEnv([sc]), sc, [E] * 4)
    assert infos[-1]["outcome"] == "collision" and infos[-1]["t"] == 1
    assert infos[-1]["robot"] == (1, 2) and infos[-1]["worker_positions"] == [(1, 2)]


def test_swap_collision():
    sc = corridor([WorkerState((1, 2), Heading.W)])
    rewards, infos, _ = run(WarehouseEnv([sc]), sc, [E])
    assert infos[-1]["outcome"] == "collision"
    assert infos[-1]["robot"] == (1, 2) and infos[-1]["worker_positions"] == [(1, 1)]
    assert rewards == [pytest.approx(-1 - 0.5 / 8)]


def test_goal_and_collision_on_the_same_step_is_collision():
    sc = corridor([WorkerState((1, 5), Heading.W)], start=(1, 4), goal=(1, 5))
    _, infos, _ = run(WarehouseEnv([sc]), sc, [E])
    assert infos[-1]["robot"] == (1, 5) and infos[-1]["outcome"] == "collision"


def test_blocked_move_leaves_robot_in_place():
    _, infos, _ = run(WarehouseEnv([corridor()]), corridor(), [N, W])
    assert [i["robot"] for i in infos] == [(1, 1)] * 3


def test_step_after_episode_end_raises():
    env = WarehouseEnv([corridor()])
    run(env, corridor(), [E] * 4)
    with pytest.raises(RuntimeError):
        env.step(WAIT)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        WarehouseEnv([corridor()]).step(WAIT)


def test_invalid_action_raises():
    env = WarehouseEnv([corridor()])
    env.reset(options={"scenario": corridor()})
    with pytest.raises(ValueError):
        env.step(5)


# --- observations and outages (SPEC §2.6, §2.7) ---


def test_observation_after_step_t_uses_schedule_entry_t():
    schedule = [False, True, True, False, False, False, False, False]
    sc = corridor(pattern="consecutive", d=2, f=0.25, schedule=schedule)
    _, infos, observations = run(WarehouseEnv([sc]), sc, [WAIT] * 8)
    valid = [float(o["scalars"][Scalar.VALID]) for o in observations]
    assert valid[:8] == [0.0 if m else 1.0 for m in schedule]
    assert [i["outage"] for i in infos[:8]] == schedule


def test_missing_first_reading_hides_the_window_at_reset():
    schedule = [True] + [False] * 7
    sc = corridor([WorkerState((1, 3), Heading.E)], pattern="scattered", f=0.125, schedule=schedule)
    obs, info = WarehouseEnv([sc]).reset(options={"scenario": sc})
    assert obs["scalars"][Scalar.VALID] == 0.0 and info["outage"] is True
    assert obs["grid"][Channel.SEEN_WORKER].sum() == 0
    assert obs["grid"][Channel.SEEN_EMPTY].sum() == 0


def test_worker_in_window_is_seen_when_sensor_is_up():
    sc = corridor([WorkerState((1, 3), Heading.E)])
    obs, _ = WarehouseEnv([sc]).reset(options={"scenario": sc})
    assert obs["grid"][Channel.SEEN_WORKER][1, 3] == 1.0


def test_observations_are_always_in_the_observation_space():
    env = WarehouseEnv([warehouse(s) for s in range(5)])
    rng = np.random.default_rng(0)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        for _ in range(100):
            obs, _, term, _, _ = env.step(int(rng.integers(5)))
            assert env.observation_space.contains(obs)
            if term:
                break


# --- reproducibility (SPEC §2.8) ---


def test_worker_trajectories_do_not_depend_on_the_controller():
    sc = warehouse(3, p=0.5)
    env = WarehouseEnv([sc])
    rng = np.random.default_rng(1)
    random_actions = [int(rng.integers(5)) for _ in range(sc.horizon)]
    _, a, _ = run(env, sc, random_actions)
    _, b, _ = run(env, sc, [WAIT] * sc.horizon)
    n = min(len(a), len(b))
    assert n > 1
    assert [i["worker_positions"] for i in a[:n]] == [i["worker_positions"] for i in b[:n]]


def test_same_scenario_replays_identically():
    sc = warehouse(4)
    env = WarehouseEnv([sc])
    actions = [E, E, S, S, E, E, S, S, E, E, E, E, WAIT, E, E, E]
    r1, i1, o1 = run(env, sc, actions)
    r2, i2, o2 = run(env, sc, actions)
    assert r1 == r2 and i1 == i2
    for x, y in zip(o1, o2, strict=True):
        assert np.array_equal(x["grid"], y["grid"]) and np.array_equal(x["scalars"], y["scalars"])


def test_episodes_before_do_not_change_worker_trajectories():
    sc = warehouse(5)
    env = WarehouseEnv([sc])
    _, fresh, _ = run(env, sc, [WAIT] * sc.horizon)
    for _ in range(3):
        run(env, warehouse(6), [E] * 5)
    _, later, _ = run(env, sc, [WAIT] * sc.horizon)
    assert fresh == later


def test_scenario_draw_follows_the_reset_seed():
    env = WarehouseEnv([warehouse(s) for s in range(10)])
    first = [env.reset(seed=7)[1]["scenario_id"]] + [
        env.reset()[1]["scenario_id"] for _ in range(5)
    ]
    second = [env.reset(seed=7)[1]["scenario_id"]] + [
        env.reset()[1]["scenario_id"] for _ in range(5)
    ]
    assert first == second and len(set(first)) > 1


# --- construction and rendering ---


def test_zero_workers_work():
    _, infos, _ = run(WarehouseEnv([corridor()]), corridor(), [E] * 4)
    assert all(i["worker_positions"] == [] for i in infos)


def test_scenarios_must_share_one_grid_shape():
    with pytest.raises(ValueError, match="grid shape"):
        WarehouseEnv([corridor(), warehouse()])
    with pytest.raises(ValueError):
        WarehouseEnv([])
    env = WarehouseEnv([corridor()])
    with pytest.raises(ValueError, match="grid shape"):
        env.reset(options={"scenario": warehouse()})


def test_ansi_render():
    sc = corridor([WorkerState((1, 3), Heading.E)])
    env = WarehouseEnv([sc], render_mode="ansi")
    env.reset(options={"scenario": sc})
    text = env.render()
    assert text.splitlines()[:3] == ["#######", "#R.W.G#", "#######"]
    assert "t=0/8" in text and "sensor=on" in text
    assert WarehouseEnv([sc]).render() is None
