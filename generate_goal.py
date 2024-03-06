import numpy as np
from get_neighbors import get_neighbors

def generate_goal(init_state, depth, puzzle_size) -> np.ndarray:
    """
    @param init_state: initial symbolic state
    @param depth: solution depth (for shortest path to goal)
    @param puzzle_size: shape of the puzzle board
    """
    if depth == 0:
        return init_state

    neighbors = get_neighbors(puzzle_size)
    goal_state = init_state.copy()
    empty_field = np.where(np.sum(init_state, axis=0) == 0)[0][0]
    banned = None
    for i in range(depth):
        # sample a piece adjacent to the empty field
        choice = neighbors[str(empty_field)]
        if banned is not None:
            idx = np.where(choice == banned)[0][0]
            choice = np.delete(choice, idx)
        next_field = np.random.choice(choice)
        # get piece that has to be moved
        box = np.where(goal_state[:, next_field] == 1)[0][0]
        goal_state[box, next_field] = 0
        goal_state[box, empty_field] = 1

        banned = empty_field
        empty_field = next_field

    return goal_state
