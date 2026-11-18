import numpy as np
import pytest

from whnav.env.grid import Action, GridMap, deadline, move_robot, shortest_path_length

# 0123
LAYOUT = [
    "....",  # 0
    ".##.",  # 1
    "....",  # 2
]


@pytest.fixture
def grid():
    return GridMap.from_strings(LAYOUT)


def test_from_strings_parses_blocked_cells(grid):
    assert (grid.height, grid.width) == (3, 4)
    assert grid.blocked[1, 1] and grid.blocked[1, 2]
    assert grid.is_free((0, 0)) and not grid.is_free((1, 1))


def test_from_strings_rejects_bad_input():
    with pytest.raises(ValueError):
        GridMap.from_strings([])
    with pytest.raises(ValueError):
        GridMap.from_strings(["...", ".."])
    with pytest.raises(ValueError):
        GridMap.from_strings([".x."])


def test_map_is_read_only(grid):
    with pytest.raises(ValueError):
        grid.blocked[0, 0] = True


def test_map_copies_its_input():
    arr = np.zeros((2, 2), dtype=bool)
    g = GridMap(arr)
    arr[0, 0] = True
    assert g.is_free((0, 0))


def test_out_of_bounds_is_not_free(grid):
    for cell in [(-1, 0), (0, -1), (3, 0), (0, 4)]:
        assert not grid.is_free(cell)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.N, (0, 0)),  # free cell above
        (Action.S, (2, 0)),  # free cell below
        (Action.E, (1, 0)),  # shelf to the east: stays
        (Action.W, (1, 0)),  # map edge to the west: stays
    ],
)
def test_move_from_left_edge(grid, action, expected):
    assert move_robot(grid, (1, 0), action) == expected


def test_move_into_shelf_stays(grid):
    assert move_robot(grid, (0, 1), Action.S) == (0, 1)


def test_move_off_map_stays(grid):
    assert move_robot(grid, (0, 0), Action.N) == (0, 0)
    assert move_robot(grid, (2, 3), Action.E) == (2, 3)


def test_wait_stays(grid):
    assert move_robot(grid, (2, 2), Action.WAIT) == (2, 2)


def test_move_from_blocked_cell_raises(grid):
    with pytest.raises(ValueError):
        move_robot(grid, (1, 1), Action.N)


def test_shortest_path_around_shelf(grid):
    assert shortest_path_length(grid, (0, 1), (2, 1)) == 4
    assert shortest_path_length(grid, (0, 0), (2, 3)) == 5
    assert shortest_path_length(grid, (2, 2), (2, 2)) == 0


def test_shortest_path_unreachable():
    g = GridMap.from_strings([".#.", ".#.", ".#."])
    assert shortest_path_length(g, (0, 0), (0, 2)) is None


def test_shortest_path_rejects_blocked_endpoints(grid):
    with pytest.raises(ValueError):
        shortest_path_length(grid, (1, 1), (0, 0))


def test_deadline_is_twice_shortest_path():
    assert deadline(5) == 10
    with pytest.raises(ValueError):
        deadline(0)
