import torch
from lib import model_pulse as model
from lib.args import args
import argparse

config_training = args['training_config']
'''
Generate pulses using a pretrained network.

params in 'args' should match 'args' in the chosen network!
you can find 'args' for the chosen network in its corresponding folder
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help='Path to the model (hint: LMB->shift+RMB->Copy as path)')
    parser.add_argument("-r", "--rounds", type=int, default=20, help="Count of rounds to perform")
    args_console = parser.parse_args()
    device = torch.device(args['environment'])
    net = model.Net(model.observation_shape, args['number_of_actions'])
    net.load_state_dict(torch.load(args_console.path))
    print(net.state_dict())
    net = net.to(device)
    for _ in range(args_console.rounds):
        r1, *_ = model.play_game(mcts_stores=None, replay_buffer=None, net=net,
                                steps_before_tau_0=0, mcts_searches=config_training['MCTS_batch_searches'],
                                mcts_batch_size=config_training['MCTS_batch_size'],
                                device=device, reward_threshold=0.9)
        if not r1:
            print('No result')