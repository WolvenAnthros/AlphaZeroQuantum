import numpy as np
import logging
"""
Dictionary of hyperparameters and configs
"""
args = {
    'training_config': {
        'play_episodes': 1,
        'MCTS_batch_searches': 10,
        'MCTS_batch_size': 6,
        'replay_buffer': 30000,  # max replay buffer size
        'learning_rate': 0.1,  # learning rate constant
        'SGD_momentum': 0.9,  # gradient descent momentum
        'training_batch_size': 256,  # number of states randomly extracted from the buffer to train Apprentice
        'training_rounds': 40,  # number of Apprentice learning rounds after every Best Player self-play
        'min_replay_to_train': 5000,  # minimal size of replay buffer to enable training
        'steps_before_tau_0': 0  # number of gamesteps with forced exploration (noise applied to action probs)
    },
    'evaluation_config': {
        'best_net_win_ratio': 0.60,  # percentage of wins for Apprentice to become new Best Player
        'num_steps_before_evaluation': 60,  # number of Best Player games before evaluation
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
        'default_pulse_list': [0, 0, 0],  # placeholder for a pulse list
        'n_dimensions': 4,  # number of system dimensions
        'num_timesteps': 100,  # number of timesteps for every pulse in pulse list
        'omega_01': 3 * 2 * np.pi,  # qbit natural frequency
        'omega_osc': 25 * 2 * np.pi,  # pulse oscillator frequency
        'amp': 3,  # pulse amplitude
        'pulse_time': 0.004,  # pulse application time
        'mu': 0.25 * 2 * np.pi,  # non-linearity coefficient
        # 'probability/leakage_reward_tradeoff': 0.9,  # excited state probability multiplier
        # 'desired_leakage': 0.0001
    },
    'pulse_array_length': 121,
    'reward_threshold': 0.001,
    'reward_threshold_to_save': 0.7,
    'reset_reward_threshold_after_eval': False,
    'number_of_actions': 3,
    'environment': 'cuda',  # cuda
    'run_name': '1/12/fidelity',
    'save_folder_name': 'saves',
    'logging_config': {
        'level': logging.DEBUG,
        'format': f'\33[1m\33[33m{"%(levelname)s | %(funcName)s: %(message)s"}\33[0m',
    }
}