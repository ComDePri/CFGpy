import numpy as np
from collections import Counter
from itertools import pairwise
from CFGpy.utils import binary_shape_to_id as bin2id
from consts import *


def preprocess_shapes(shapes):
    return np.array(shapes)


def get_steps(actions: np.ndarray):
    shape_ids = actions[:, SHAPE_ID_IDX]
    steps = list(pairwise(shape_ids))
    return steps


def get_all_players_steps(all_players_data):
    all_steps = []
    for player_data in all_players_data:
        steps = get_steps(preprocess_shapes(player_data["actions"]))
        all_steps.extend(steps)

    return all_steps


def get_step_orig_map(all_players_data, alpha=0):
    """
    Returns a mapping from each step in the sample to its originality value. The originality value of a step denoted
    (s1, s2) is given by -log10(p) where p is the transition probability P(s2 | s1), estimated over the given sample.
    :param all_players_data: parsed CFG data loaded from json.
    :param alpha: pseudocount for Laplace smoothing (see https://en.wikipedia.org/wiki/Additive_smoothing).
    :return: a dict of the form {(int, int): float}
    """
    all_steps = get_all_players_steps(all_players_data)
    step_counter = Counter(all_steps)
    # TODO: step_counter is already present MeasureCalculator before calling this, so its calculation here is
    #  somewhat redundant. on the other hand, getting it as a parameter seems like bad design since its directly
    #  calculated from a different parameter. This issue is bypassed if this function becomes a method of
    #  MeasureCalculator and can access its all_steps_counter member.

    # calculate total number of steps that started from each shape
    total_steps_from_shape = Counter()
    for player_data in all_players_data:
        actions = player_data["actions"]
        step_starter_ids = [bin2id(shape[SHAPE_ID_IDX]) for shape in actions[:-1]]  # last shape is not a step starter
        total_steps_from_shape += Counter(step_starter_ids)

    # transform counts to orig values: -log10 of alpha-smoothed probability of the step (second shape given first)
    step_orig_map = {}
    for step, count in step_counter.items():
        total = total_steps_from_shape[step[0]]
        smoothed_step_probability = (count + alpha) / (total + len(total_steps_from_shape) * alpha)
        step_orig_map[step] = -np.log10(smoothed_step_probability)

    return step_orig_map
