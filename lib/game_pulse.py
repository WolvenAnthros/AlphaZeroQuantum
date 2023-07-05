import numpy as np

import old_simspeed
from lib.args import args
from simulation_speed import reward_calculation
#from old_simspeed import probability_calculation
import random

max_sequence_length = args['pulse_array_length']
# deprecated, string representation
# INITIAL_STATE = ''.join([str(int(elem)) for elem in np.zeros(MAX_PULSE_LENGTH)])  # initial state is an array of 0's\
# -> a string of '0000...0', tight form
INITIAL_STATE = np.zeros(max_sequence_length, dtype=int)
INITIAL_STATE = tuple(INITIAL_STATE)
INITIAL_INDEX = 0

config = args['qbit_simulation_config']
reward_threshold = args['reward_threshold']


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
    random.seed(31)
    indices = [x for x in range(max_sequence_length-1)]
    random.shuffle(indices)
    index = indices[idx]

    state_array[index] = action - 1  # if action=0 -> 0-1 = -1 -> actual action, same with all actions (0,1,2)
    state_array[-1] = idx  # for effective state storing

    #if reward > 0.47:
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
