import numpy as np


def state_to_symbolic_obs(img) -> [bool]:
    """
    maps state of the world to symbolic observations given an image of the scene
    :param img: input image
    :return: mapping, boolean array (which stone is on which field)
    """
    obs = np.array((6, 6), dtype=bool)

    return obs
