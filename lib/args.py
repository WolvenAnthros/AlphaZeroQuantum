import numpy as np
import logging
from datetime import datetime

date = datetime.now()
"""
Dictionary of hyperparameters and configs
"""
args = {
    'training_config': {
        'play_episodes': 1,
        'MCTS_batch_searches': 15,
        'MCTS_batch_size': 10,
        'replay_buffer': 30000,  # max replay buffer size
        'learning_rate': 0.1,  # learning rate constant
        'SGD_momentum': 0.9,  # gradient descent momentum
        'training_batch_size': 256,  # number of states randomly extracted from the buffer to train Apprentice
        'training_rounds': 20,  # number of Apprentice learning rounds after every Best Player self-play
        'min_replay_to_train': 4600,  # minimal size of replay buffer to enable training
        'steps_before_tau_0': 5  # number of gamesteps with forced exploration (noise applied to action probs)
    },
    'evaluation_config': {
        'best_net_win_ratio': 0.60,  # percentage of wins for Apprentice to become new Best Player
        'num_steps_before_evaluation': 100,  # number of Best Player games before evaluation
        'evaluation_rounds': 10  # number of games between Apprentice and Best Player
    },
    'MCTS_config': {
        'c_puct': 1,  # exploration/exploitation tradeoff MCTS coefficient
        'tau': 1,  # default value of tau coefficient (helps for exploration of states)
    },
    'convNet_config': {
        'num_filters': 64,  # number of convolutional filters
        'conv_layers_kernel_size': 3,  # convolutional layers kernel size
        'conv_layers_padding': 1  # padding value
    },
    'qbit_simulation_config': {
        'n_dimensions': 3,  # number of system dimensions
        'num_timesteps': 1000,  # number of timesteps for every pulse in pulse list
        'omega_01': 4.54643 * 2 * np.pi,  # * 1e9,  # qbit natural frequency
        'omega_osc': 25 * 2 * np.pi,  # * 1e9,  # pulse oscillator frequency
        'amp': 4,  # * 1e-25,  # pulse amplitude
        'pulse_time': 0.004,  # * 1e-9,  # pulse application time
        'mu': 0.25 * 2 * np.pi,  # * 1e9,  # non-linearity coefficient
        'theta': 0.034,  # pulse rotation angle
        'example_scallop': '11100111000110000000011100000001110001110001' * 6,
        'probability/leakage_reward_tradeoff': 0.9,  # excited state probability multiplier
        # 'desired_leakage': 0.0001
    },
    'pulse_array_length': 130,
    'reward_threshold': 0.001,
    'reward_threshold_to_save': 0.5,
    'reset_reward_threshold_after_eval': False,
    'number_of_actions': 3,
    'environment': 'cuda',  # cuda
    'run_name': f'{date.day}_{date.month}_{date.year}_standard/fidelity_check',
    'save_folder_name': 'saves',
    'enable_highlight': False,
    'logging_config': {
        'level': logging.DEBUG,
        'format': f'\33[1m\33[33m{"%(levelname)s | %(funcName)s: %(message)s"}\33[0m',
    }
}