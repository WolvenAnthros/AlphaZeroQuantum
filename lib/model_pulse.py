# collection for deque replay buffer
import collections
# math and logging
import os
import numpy as np
# logging module
import probability_calc
from logger import logger as logs
# conv NNet is written on Pytorch
import torch
import torch.nn as nn
# get config and parameters
from lib.args import args
# game and MCTS

from lib import mcts_pulse as mcts

if args['gates_computing']:
    import gate_calculation as game
else:
    from lib import game_pulse as game
    from simulation_speed import reward_calculation

# initialize parameters for body convolutional layers
observation_shape = (args['pulse_array_length'],)
gates = args['gates_computing']

conv_kernel_size = args['convNet_config']['conv_layers_kernel_size']
conv_layers_padding = args['convNet_config']['conv_layers_padding']
conv_num_filters = args['convNet_config']['num_filters']
conv_padding = args['convNet_config']['conv_layers_padding']
input_layer_size = 600
layer_size = 400


class Net(nn.Module):
    """
    Convolutional Neural Network
    Consists of several convolutional body layers and two heads:
    1. value head (outputs predicted value for the current state)
    2. policy head (outputs predicted action probabilities for the current state)
    """

    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features=args['pulse_array_length'], out_features=input_layer_size),
            nn.BatchNorm1d(input_layer_size),
            nn.LeakyReLU()
        )
        # simplified residual layers
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.BatchNorm1d(layer_size),
            nn.LeakyReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.BatchNorm1d(layer_size),
            nn.LeakyReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.BatchNorm1d(layer_size),
            nn.LeakyReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.BatchNorm1d(layer_size),
            nn.LeakyReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.BatchNorm1d(layer_size),
            nn.LeakyReLU()
        )
        # value head ( convolutional layer -> linear layer -> output )
        self.value = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=120),
            # please consider putting all of these params into a single dict
            nn.LeakyReLU(),
            nn.Linear(in_features=120, out_features=1),
            nn.Tanh()
        )
        # policy head
        self.policy = nn.Sequential(
            nn.Linear(in_features=layer_size, out_features=actions_n),
            nn.Sigmoid()
        )

    def _get_conv_val_size(self,
                           shape):  # reshape convolutional
        o = self.conv_value(torch.zeros(1, *shape))
        return int(np.prod(o.size()))  # return product of convolutional layer size

    def _get_conv_policy_size(self, shape):  # same again. why twice?
        o = self.conv_value(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        v = self.input_layer(x)
        v = v + self.layer_1(v)
        v = v + self.layer_2(v)
        v = v + self.layer_3(v)
        v = v + self.layer_4(v)
        v = v + self.layer_5(v)
        value = self.value(v)
        policy = self.policy(v)
        return policy, value


def state_lists_to_batch(state_lists, device='cpu'):
    """
    Convert list of states to batch (input) for NNet
    :param device: cuda/cpu
    :param state_lists: list of states
    :return Variable with observations
    """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)  # size of batch
    batch = np.zeros((batch_size,) + observation_shape,
                     dtype=np.float)  # an array of zeros with the shape of (batch_size,1,PULSE_LENGTH)
    for idx, state in enumerate(state_lists):
        if gates:
            state = state[:game.game_length]  # FIXME: state truncation
        state_array = np.array(state, dtype=int)
        batch[idx] = state_array
    # create an array based on a string, separator is a whitespace pytotch Tensor is a ndmatrix, so it is inferred
    # that batch should  have the dimensionality of [state_lists,1,PULSE_LENGTH]
    return torch.tensor(batch).to(device,
                                  dtype=torch.float32)  # 'batch' is what the NN will receive
    # dtype is mentioned because dtype in input and weights of a layer should match


def play_game(mcts_stores, replay_buffer, net, steps_before_tau_0,
              mcts_searches, mcts_batch_size, device='cpu', reward_threshold=0, enable_highlight=True):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_searches: number of MCTS batch searches
    :param mcts_batch_size: number of MCT searches in every MCTS batch
    :param device: CPU or CUDA
    :param steps_before_tau_0: number of steps before determenistic approach begins (stochastic approach at first)
    :param mcts_stores: could be None or MCTS
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net: current NNet (represents a player)
    :param enable_highlight: enable/disable printing out pulse array
    :param reward_threshold: current reward threshold for a pulse array to be passed to the replay buffer
    :return: value for the game (result of the qubit transition simulation)
    """
    # perform ordinary checks
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:  # if we don't have an instance of MCTS
        mcts_stores = mcts.MCTS()  # create a couple of MCTS for both "players"
    elif isinstance(mcts_stores, mcts.MCTS):  # in other case
        mcts_stores = mcts_stores  # fill it with saved results

    state = game.INITIAL_STATE  # initialize the game state
    current_index = game.INITIAL_INDEX  # initialize the index

    step = 0  # initialize steps befofe deterministic approach (tau=0)
    tau = 1 if steps_before_tau_0 > 0 else 0  # to not let our noise interfere the final picture
    game_history = []

    result = None
    while result is None:  # until we find a reward
        mcts_stores.search_batch(
            # perform a batch of searches (consisting of several minibatches, see MCTS)
            count=mcts_searches,
            batch_size=mcts_batch_size,
            state=state,
            index=current_index,
            net=net,
            device=device,
            reward_threshold=reward_threshold
        )

        # extract the best actions based on their visit times or on distribution + visit times (see MCTS)
        probs, _ = mcts_stores.get_policy_value(state=state, tau=tau)
        game_history.append((state, probs))  # game history saving
        action = np.random.choice(mcts.action_num, p=probs)
        if action not in game.allowed_moves(state):
            logs.critical(f'Prohibited move: {action}')
        # # execute a game move
        # show_action = game.operations_list[action]()
        # game.operation_history.append(str(show_action))
        state, reward, done = game.move(state=state, idx=current_index, action=action)  # get reward and next state
        if args['gates_computing']:
            result_game_state, result_quantum_state = game.decode(state)
            result_game_state = list(result_game_state)
            for idx, elem in enumerate(result_game_state[:current_index + 1]):
                result_game_state[idx] = game.operations_list[elem].__name__

        # pass the game states in replay buffer only if the final state reward has exceeded the current reward threshold
        if done:  # reward > reward_threshold or
            result_show = 0
            if not args['gates_computing']:
                result_state = state
                result_show = reward_calculation(pulse_list=result_state)
                result_state = list(result_state)

            # for elem in result_state[:re]

            probability = 0
            probability = probability_calc.probability_calculation(result_state)

            # logs.debug(f'Operations: {game.operation_history}')
            # game.operation_history = []
            result = reward

            if enable_highlight:
                logs.debug(f'State:  {result_state}')
            if True:  # reward > reward_threshold
                reward_threshold = reward
                if not enable_highlight:
                    logs.debug(f'State: {result_state}')
                if replay_buffer is not None:
                    for state, probs in reversed(game_history):
                        replay_buffer.append(
                            (state, probs, result)
                        )
            # save the best pulse lists into a separate txt file
            if result_show > args['reward_threshold_to_save']:
                result_state = result_state
                saves_path = os.path.join(args['save_folder_name'], args['run_name'])
                best_pulses_save_path = os.path.join(saves_path, 'Pulses_evolution.txt')
                with open(best_pulses_save_path, 'a') as file:
                    file.write(f'State: {result_state} \n Result: {result} \n')

            logs.debug(f'Fidelity: {result_show:.4f}, True reward:{result:.4f}, Proba: {probability:.3f}')
            break
            # Index residual:{idx_residual*(current_index/lib.game_pulse.MAX_PULSE_LENGTH):.3g},
        current_index += 1  # after a move is executed, go to the next index
        step += 1  # increment tau steps
        if step >= steps_before_tau_0:
            tau = 0
    return result, step, reward_threshold
