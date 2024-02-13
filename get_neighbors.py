
def get_neighbors(puzzle_size):
    """
    Calculates the neibhors of each field, depending on the puzzle size
    """
    neighborlist = {}
    for i in range(puzzle_size[0]):
        for j in range(puzzle_size[1]):
            # look on which side we have a neighbor and mark it in array
            neighbors = []
            field = j + (i * puzzle_size[1])
            if i != 0:
                # a neighbor above
                neighbors.append(field - puzzle_size[1])
            if j != 0:
                # no neighbor on left
                neighbors.append(field - 1)
            if j != puzzle_size[1] - 1:
                # a neighbors on right
                neighbors.append(field + 1)
            if i != puzzle_size[0] - 1:
                # a neighbor below
                neighbors.append(field + puzzle_size[1])

            neighborlist[str(j + (i * puzzle_size[1]))] = neighbors

    return neighborlist
