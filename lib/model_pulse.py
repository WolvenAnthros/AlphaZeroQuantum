# collection for deque replay buffer
import collections
# math and logging
import os
import numpy as np
# logging module
from logger import logger as logs
# conv NNet is written on Pytorch
import torch
import torch.nn as nn
# get config and parameters
from lib.args import args
# game and MCTS
from lib import mcts_pulse as mcts
from lib import game_pulse as game

# initialize parameters for body convolutional layers
observation_shape = (1, args['pulse_array_length'])

conv_kernel_size = args['convNet_config']['conv_layers_kernel_size']
conv_layers_padding = args['convNet_config']['conv_layers_padding']
conv_num_filters = args['convNet_config']['num_filters']
conv_padding = args['convNet_config']['conv_layers_padding']


class Net(nn.Module):
    """
    Convolutional Neural Network
    Consists of several convolutional body layers and two heads:
    1. value head (outputs predicted value for the current state)
    2. policy head (outputs predicted action probabilities for the current state)
    """

    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()
        # convolutional input layer
        self.conv_input = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),
            # input_shape[0] stands for OBSERVATION_SHAPE[0] since convNet accepts data BY CHANNELS
            # padding and stride = 1 by default
            nn.BatchNorm1d(conv_num_filters),  # BatchNormalization for every feature map
            nn.LeakyReLU()  # LeakyReLU activation function
        )
        # simplified residual layers
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),  # input dim = output dim
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),  # input dim = output dim
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),  # input dim = output dim
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),  # input dim = output dim
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=conv_num_filters, kernel_size=conv_kernel_size,
                      padding=conv_padding),  # input dim = output dim
            nn.BatchNorm1d(conv_num_filters),
            nn.LeakyReLU()
        )
        # reshape the input for value/policy head (ResNet peculiar feature)
        body_out_shape = (conv_num_filters,) + input_shape[1:]

        # value head ( convolutional layer -> linear layer -> output )
        self.conv_value = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=1, kernel_size=1),
            # single feature map with kernel = 1
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )
        conv_value_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(in_features=conv_value_size, out_features=20),
            # please consider putting all of these params into a single dict
            nn.LeakyReLU(),
            nn.Linear(in_features=20, out_features=1),
            nn.Tanh()
        )

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv1d(in_channels=conv_num_filters, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(in_features=conv_policy_size, out_features=actions_n)
        )

    def _get_conv_val_size(self,
                           shape):  # reshape convolutional
        o = self.conv_value(torch.zeros(1, *shape))
        return int(np.prod(o.size()))  # return product of convolutional layer size

    def _get_conv_policy_size(self, shape):  # same again. why twice?
        o = self.conv_value(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Technical implementation of the NNet, part of Pytorch syntax
        """
        batch_size = x.size()[0]
        v = self.conv_input(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        v = v + self.conv_6(v)
        value = self.conv_value(v)  # depends on body layers
        value = self.value(value.view(batch_size,
                                      -1))  # see nn.Tensor.view() documentation
        policy = self.conv_policy(v)  # depends on body layers
        policy = self.policy(policy.view(batch_size, -1))
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
        state_array = np.frombuffer(state, dtype=int)
        batch[idx] = state_array
        # string representation, deprecated
        # state_array = np.array(re.findall(r'(?<!\s)(?<!\s\d)(?<!\s\d{2})[+-]?\d', state),dtype=int) # tight printing
        # state_array = np.array(re.findall(r'[+-]?\d', state), dtype=int)
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
    current_index = 0  # initialize the index

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

        # execute a game move
        state, reward, done = game.move(state=state, idx=current_index, action=action)  # get reward and next state

        # pass the game states in replay buffer only if the final state reward has exceeded the current reward threshold
        if reward > reward_threshold or done or reward > 1:
            result = reward
            if reward > reward_threshold:
                logs.debug(f'State:  {np.frombuffer(state, dtype=int)}')
                reward_threshold = reward
                # train.reward_threshold = reward
                if replay_buffer is not None:
                    for state, probs in reversed(game_history):
                        replay_buffer.append(
                            (state, probs, result)
                        )
            # save the best pulse lists into a separate txt file
            if result > args['reward_threshold_to_save']:
                saves_path = os.path.join(args['save_folder_name'], args['run_name'])
                best_pulses_save_path = os.path.join(saves_path, 'best_pulses.txt')
                with open(best_pulses_save_path, 'a') as file:
                    file.write(f'State: {np.frombuffer(state, dtype=int)} \n Result: {result} \n')
            if enable_highlight:
                logs.debug(f'State:  {np.frombuffer(state, dtype=int)}')
            logs.debug(f'Result: {result:.3f}, Infidelity: {1-result:.2e}')
            break

        current_index += 1  # after a move is executed, go to the next index
        step += 1  # increment tau steps
        if step >= steps_before_tau_0:
            tau = 0
    return result, step, reward_threshold
