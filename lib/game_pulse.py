import numpy as np
from lib.args import args
from simulation_speed import reward_calculation

max_sequence_length = args['pulse_array_length']
config = args['qbit_simulation_config']
reward_threshold = args['reward_threshold']
polarities_num = args['number_of_polarities']

unexplored_pulse = 9
INITIAL_STATE = np.full(max_sequence_length, unexplored_pulse, dtype=int)
INITIAL_STATE = tuple(INITIAL_STATE)
INITIAL_INDEX = 0
initial_actions_list = np.array([x for x in range(polarities_num * max_sequence_length)])


def allowed_moves(state):
    state_array = np.array(state)
    actions_list = initial_actions_list.copy()
    for pulse_idx, pulse in enumerate(state_array):
        action_idx = pulse_idx * polarities_num
        if pulse != unexplored_pulse:
            actions_list[action_idx:action_idx + polarities_num] = -1
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

    # decode action number into the action type and its index
    index = action // polarities_num
    true_action = action % polarities_num

    state_array[index] = true_action - 1  # if action=0 -> 0-1 = -1 -> actual action

    reward = reward_calculation(pulse_list=state_array)
    reward = 1 - ((1 - reward) / 1) ** 0.5
    reward = round(reward, 4)

    done = False
    if idx == max_sequence_length - 1:  # or probability > 0.5)
        done = True

    state = tuple(state_array)

    return state, reward, done


if __name__ == '__main__':
    state = INITIAL_STATE
    action = 1
    print(move(state, INITIAL_INDEX, action))
    state, _, _ = move(INITIAL_STATE, 125, 375)
    print(allowed_moves(state))
