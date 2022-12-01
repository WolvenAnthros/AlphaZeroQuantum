import numpy as np
from lib.args import args
from simulation_speed import reward_calculation

MAX_PULSE_LENGTH = args['pulse_array_length']
# deprecated, string representation
# INITIAL_STATE = ''.join([str(int(elem)) for elem in np.zeros(MAX_PULSE_LENGTH)])  # initial state is an array of 0's\
# -> a string of '0000...0', tight form
INITIAL_STATE = np.zeros(MAX_PULSE_LENGTH, dtype=int).tobytes()
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
    assert isinstance(idx, int)
    assert idx < MAX_PULSE_LENGTH
    state_array = np.frombuffer(state, dtype=int).copy()  # convert bytes to array
    state_array[idx] = action - 1  # if action=0 -> 0-1 = -1 -> actual action, same with all actions (0,1,2)
    state_array[-1] = idx  # for effective state storing
    reward = reward_calculation(pulse_list=state_array)
    # reward = reward if reward > reward_threshold else 0 # reward threshold moved to model_pulse
    done = False
    if idx == MAX_PULSE_LENGTH - 2:
        done = True
    state = state_array.tobytes()

    return state, reward, done


# is here for future decorative purposes
def render(state):
    return list(state)
