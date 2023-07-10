import numpy as np

import old_simspeed
from lib.args import args
from simulation_speed import reward_calculation
# from old_simspeed import probability_calculation
import random

max_sequence_length = args['pulse_array_length']
config = args['qbit_simulation_config']
reward_threshold = args['reward_threshold']
polarities_num = args['number_of_polarities']
# deprecated, string representation
# INITIAL_STATE = ''.join([str(int(elem)) for elem in np.zeros(MAX_PULSE_LENGTH)])  # initial state is an array of 0's\
# -> a string of '0000...0', tight form
INITIAL_STATE = np.zeros(max_sequence_length, dtype=int)
#INITIAL_STATE = np.array([1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, 0, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 1, 1, 1, -1, -1, 0, -1, 0, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 0, -1, -1, -1, -1, 0, 1, 1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 0, -1, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1])

INITIAL_STATE = tuple(INITIAL_STATE)
INITIAL_INDEX = 0


def allowed_moves(state):
    state_array = np.array(state)
    state_array, state_idx = state_array[:-1], state_array[-1]
    actions_list = np.array([x for x in range(polarities_num * (max_sequence_length - 1))])

    for pulse_idx, pulse in enumerate(state_array):
        action_idx = pulse_idx * polarities_num
        if pulse != 0:
            actions_list[action_idx:action_idx + polarities_num] = -1

    state_idx = state_idx * polarities_num if state_idx != 0 else max_sequence_length*polarities_num # FIXME: state_idx = 0 leads to process halting!

    actions_list[state_idx:state_idx + polarities_num] = -1

    return actions_list


def move(state, idx, action):
    """
    Perform pulse at a given index in a given state
    :param state: pulse array in string form (with index)
    String form makes states hashable, so they can be used in replay buffer
    :param idx: index in pulse list where to make a pulse
    :param action: possible action (0,1,2 by default)
    :return: tuple of (state, reward), consisting of new state and reward from 0 to 1
    """

    state_array = np.array(state)
    # random.seed(42)
    # indices = [x for x in range(max_sequence_length - 1)]
    # random.shuffle(indices)
    # index = indices[idx]

    index = action // polarities_num
    true_action = action % polarities_num

    state_array[index] = true_action - 1  # if action=0 -> 0-1 = -1 -> actual action, same with all actions (0,1,2)
    state_array[-1] = index #idx or index?  # for effective state storing

    # if reward > 0.47:
    reward = reward_calculation(pulse_list=state_array)
    reward = 1 - ((1 - reward) / 1) ** 1
    reward = round(reward, 4)

    done = False
    if idx == max_sequence_length - 2:  # or probability > 0.5)
        done = True
    state = tuple(state_array)

    return state, reward, done


# is here for future decorative purposes
def render(state):
    return list(state)


if __name__ == '__main__':
    state = INITIAL_STATE
    action = 1
    print(move(state, INITIAL_INDEX, action))
    state, _, _ = move(INITIAL_STATE, 125, 375)
    print(allowed_moves(state))
