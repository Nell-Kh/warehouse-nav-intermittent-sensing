import pytest

from whnav.env.rules import Outcome, is_collision, step_outcome, step_reward

# --- collisions (SPEC §2.4) ---


def test_same_cell_is_collision():
    assert is_collision((0, 0), (0, 1), [(0, 2)], [(0, 1)])


def test_swap_is_collision():
    assert is_collision((0, 0), (0, 1), [(0, 1)], [(0, 0)])


def test_following_is_not_collision():
    # worker moves into the cell the robot just left; robot moved on
    assert not is_collision((0, 1), (0, 2), [(0, 0)], [(0, 1)])


def test_robot_waits_worker_walks_away():
    assert not is_collision((1, 1), (1, 1), [(1, 2)], [(1, 3)])


def test_robot_waits_worker_walks_in():
    assert is_collision((1, 1), (1, 1), [(1, 2)], [(1, 1)])


def test_workers_may_share_a_cell():
    assert not is_collision((0, 0), (0, 0), [(2, 2), (2, 3)], [(2, 3), (2, 3)])


def test_any_one_worker_triggers_collision():
    assert is_collision((0, 0), (0, 1), [(5, 5), (0, 1)], [(5, 6), (0, 0)])


def test_no_workers_no_collision():
    assert not is_collision((0, 0), (0, 1), [], [])


def test_mismatched_worker_lists_raise():
    with pytest.raises(ValueError):
        is_collision((0, 0), (0, 1), [(1, 1)], [])


# --- outcomes (SPEC §2.4) ---


def test_collision_beats_goal_on_same_step():
    assert step_outcome((3, 3), (3, 3), collided=True, t=2, horizon=10) is Outcome.COLLISION


def test_goal_on_last_step_is_success():
    assert step_outcome((3, 3), (3, 3), collided=False, t=10, horizon=10) is Outcome.SUCCESS


def test_deadline_on_last_step():
    assert step_outcome((0, 0), (3, 3), collided=False, t=10, horizon=10) is Outcome.DEADLINE


def test_running_before_deadline():
    out = step_outcome((0, 0), (3, 3), collided=False, t=9, horizon=10)
    assert out is Outcome.RUNNING and not out.is_terminal


def test_step_outside_horizon_raises():
    with pytest.raises(ValueError):
        step_outcome((0, 0), (3, 3), collided=False, t=11, horizon=10)
    with pytest.raises(ValueError):
        step_outcome((0, 0), (3, 3), collided=False, t=0, horizon=10)


# --- rewards and episode returns (SPEC §2.5) ---


def episode_return(final: Outcome, k: int, horizon: int) -> float:
    running = [step_reward(Outcome.RUNNING, horizon)] * (k - 1)
    return sum(running) + step_reward(final, horizon)


@pytest.mark.parametrize("k", [1, 5, 20])
def test_success_return(k):
    assert episode_return(Outcome.SUCCESS, k, 20) == pytest.approx(1 - 0.5 * k / 20)


@pytest.mark.parametrize("k", [1, 5, 20])
def test_collision_return(k):
    assert episode_return(Outcome.COLLISION, k, 20) == pytest.approx(-1 - 0.5 * k / 20)


def test_deadline_return_is_minus_half():
    assert episode_return(Outcome.DEADLINE, 20, 20) == pytest.approx(-0.5)


def test_return_ordering_holds_for_every_step():
    horizon = 30
    deadline_return = episode_return(Outcome.DEADLINE, horizon, horizon)
    for k in range(1, horizon + 1):
        assert episode_return(Outcome.SUCCESS, k, horizon) > deadline_return
        assert deadline_return > episode_return(Outcome.COLLISION, k, horizon)
