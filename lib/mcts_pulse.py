# game and the convolutional NNet
from logger import logger as logs
from lib import model_pulse as model
# math libraries
import math as m
import numpy as np
# torch for softmax functions
import torch.nn.functional as F
# get config and parameters
from lib.args import args
if args['gates_computing']:
    import gate_calculation as game
else:
    from lib import game_pulse as game

config = args['MCTS_config']
action_num = args['number_of_actions']


class MCTS:
    """
    Monte-Carlo Tree Search
    Class keeps statistics for every state encountered during the search
    """

    def __init__(self, c_puct=config['c_puct']):
        """
        Initiates MCTS parameter storages for the current search
        :parameter c_puct: exploration term for UCB score
        """
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's act, state_int -> [V(s, a)]
        self.value = {}
        # value of actions divided by their visit count, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}

    def clear(self):
        """
        Clears MCTS parameters without deleting an instance of the class
        """
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state, index, reward_threshold=0):  # START FROM CURRENT INDEX -> MOVE UNTIL LEAF NODE INDEX
        """
        Traverse the tree until the end of game or leaf node
        Starting at current index --> Move until leaf node index

        :param reward_threshold: current reward threshold for a state to be passed into the buffer
        :param state: current pulse array state
        :param index: index at which action is taken
        :return: tuple of (value, current_state, current_index, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome at the leaf
        2. leaf_state: leaf state
        4. states: list of states traversed
        5. actions: list of actions taken
        """
        states = []  # to keep track of states passed
        actions = []  # to keep track of actions made
        current_state = state
        current_index = index
        value = None

        # We traverse the game tree until we find a game end/an unexplored leaf node
        while current_state in self.probs:
            states.append(current_state)

            action_visits = self.visit_count[current_state]  # list of action visits for current state
            total_visits_sqrt = m.sqrt(sum(action_visits))  # sqrt of visit count for all actions in a state
            probs = self.probs[current_state]  # probabilities of actions for current state
            actions_avg_values = self.value_avg[current_state]  # values (average) for all actions in a state

            # in the root node we want to add Dirichlet noises to action probabilities
            if state == current_state:  # root node
                noises = np.random.dirichlet([0.03] * action_num)
                probs = [  # adding noise to probabilities
                    0.75 * prob + 0.25 * noise
                    for prob, noise in zip(probs, noises)
                ]
            # calculate UCB score for every action in every node we meet
            ucb_score = [
                value + self.c_puct * prob * total_visits_sqrt / (1 + count)  # Q + C * P * sqrt(N/1+n)
                for value, prob, count in
                zip(actions_avg_values, probs, action_visits)
            ]
            # masking prohibited moves
            action_sequence = game.initial_actions_list.copy()
            action_sequence = np.append(action_sequence, -1) # -1 means invalid action
            prohibited_actions = set(action_sequence) - set(game.allowed_moves(current_state))
            '''
            If there is only -1 in prohibited actions list, it means that all actions are possible
            '''
            for prohibited in prohibited_actions:
                ucb_score[prohibited] = -np.inf

            action = int(np.argmax(ucb_score))  # we always choose an action based entirely on UCB score
            # argmax returns the indices of the maximum values along an axis (list in our case), \
            # so we should name our actions as (0,1,2)
            actions.append(action)  # save action

            current_state, reward, done = game.move(
                state=current_state,
                idx=current_index,
                action=action
            )

            # if our reward is non-zero, which means that the game is ended
            if done:  # REMIND: mcts reward>reward threshold  reward > reward_threshold or
                value = reward

            current_index += 1  # index extension

            if current_index % 250 == 0:
                logs.critical(f'State: {current_state}, \n'
                              f' action {action}, current_index {current_index}, prohibited: {prohibited_actions} ')

        return value, current_state, current_index, states, actions

    def search_batch(self, count, batch_size, state,
                     index, net, device="cpu", reward_threshold=0):
        """
        Performs a batch of searches consisting of several minibatches

        :param reward_threshold: current reward threshold for a state to be passed into the buffer
        :param count: total number of minibatch searches
        :param batch_size: number of leaf state searches during each batch
        :param state: Current game state
        :param index: current pulse index
        :param net: Neural Network which is currently used
        :param device: GPU or CUDA
        :return:
        """
        for _ in range(count):
            self.search_minibatch(batch_size, state,
                                  index, net, device, reward_threshold)

    def search_minibatch(self, num_searches, state, index, net, device='cpu',
                         reward_threshold=0):  # CURRENT INDEX -> MANY LEAF INDICES
        """
        Performs several MCTS leaf searches
        Does not return anything, updates MCTS parameters

        :param reward_threshold: current reward threshold for a state to be passed into the buffer
        :param num_searches: number of leaf state searches
        :param state: current state
        :param index: current pulse index
        :param net: neural network (Apprentice network in our case)
        :param device: CPU or CUDA
        """
        backpropagation_queue = []
        expand_states = []
        expand_indices = []
        expand_queue = []
        planned = set()  # planned for expansion
        # perform several MCTS searches
        for _ in range(num_searches):
            value, leaf_state, leaf_index, states, actions = self.find_leaf(state=state, index=index,
                                                                            reward_threshold=reward_threshold)
            # logs.info(f'\nLeaf index: {leaf_index}, '
            #           f'actions = {actions}, '
            #           f'leaf value = {value}, '
            #           f'\n leaf state {leaf_state} '
            #           f'\n initial state{state}')

            if value is not None:  # if we reached the end of the game
                backpropagation_queue.append((value, states, actions))  # save the result for backpropagation

            else:  # if we discover unexplored node
                if leaf_state not in planned:  # if the node is not in planned for expansion
                    planned.add(leaf_state)  # add to planned states
                    expand_states.append(leaf_state)  # schedule the expand process for this state
                    expand_indices.append(leaf_index)  # and index
                    expand_queue.append((leaf_state, states,
                                         actions))  # schedule the expand process for combined parameters
        # To expand, we convert the states into the form required by the model
        # and ask our network to return prior
        # probabilities and values for the batch of states
        # We will use those probabilities to
        # create nodes, and the values will be backed up on a final statistics update

        # perform expansion of nodes
        if expand_queue:
            # form a PyTorch tensor of states and pass it to our net
            batch_v = model.state_lists_to_batch(
                state_lists=expand_states,
                device=device
            )
            # let the net predict action probabilities and values
            logits_v, values_v = net(batch_v) # REMIND: grad_fn differs from the original (SigmoidBackward vs Adam)

            probs_v = F.softmax(logits_v, dim=1)
            # after prediction, convert them to numpy for further actions
            values = values_v.data.cpu().numpy()[0]  # REMIND: changed values interpretation from [[]] to []
            probs = probs_v.data.cpu().numpy()
            # then we create the nodes
            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0] * args['number_of_actions']
                self.value[leaf_state] = [0.0] * args['number_of_actions']
                self.value_avg[leaf_state] = [0.0] * args['number_of_actions']
                self.probs[leaf_state] = prob
                backpropagation_queue.append((value, states, actions))

        # perform backpropagation
        for value, states, actions in backpropagation_queue:
            # it's the BACKpropagation so we travel through states and actions backwards
            for state, action in zip(states[::-1],
                                     actions[::-1]):
                self.visit_count[state][action] += 1  # update visit count
                self.value[state][action] += value  # update value
                self.value_avg[state][action] = self.value[state][action] / self.visit_count[state][
                    action]  # update avg values

    def get_policy_value(self, state, tau=config['tau']):
        """
        Extract policy and action-values by the state
        :param tau: exploration/exploitation switch parameter
        :param state: current state
        :return: probabilities for actions P(a) and avg values Q(a)
        """
        counts = self.visit_count[state]  # counts of actions visited
        if tau == 0:
            probs = [0.0] * args['number_of_actions']
            probs[np.argmax(counts)] = 1.0  # we choose the most visited action (determenistic approach)
        else:
            counts = [count ** (1.0 / tau) for count in counts]  # distribution for improved exploration
            total = sum(counts)  # total sum of counts for all actions
            probs = [count / total for count in counts]  # normalized "probabilities" of actions based on visit count
        values = self.value_avg[state]  # avg values (divided by their visit count?)
        return probs, values
