import numpy as np
from lib.args import args
from simulation_speed import reward_calculation

MAX_PULSE_LENGTH = args['max_subarray_length']
INITIAL_STATE = np.zeros(MAX_PULSE_LENGTH, dtype=int).tobytes()

config = args['qbit_simulation_config']
reward_threshold = args['reward_threshold']

def move(state, idx, action, subarray_length):
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
    subarray_length += 1
    state_array = np.frombuffer(state, dtype=int).copy()  # convert bytes to array
    state_array = state_array[:args['max_subarray_length']]
    num_repetitions = args['pulse_array_length']/subarray_length
    state_array[idx] = action - 1  # if action=0 -> 0-1 = -1 -> actual action, same with all actions (0,1,2)
    long_state_array = np.tile(state_array[:subarray_length], int(num_repetitions))
    reward = reward_calculation(pulse_list=long_state_array)
    done = False
    if idx == MAX_PULSE_LENGTH - 1:
        done = True
    state = state_array.tobytes()

    return state, reward, done, subarray_length
