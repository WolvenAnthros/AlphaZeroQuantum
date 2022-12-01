# logs
from logger import logger as logs
# to create the save folder
import os
# to measure the time taken for the run
import time
# agent/play tracker library
import ptan
import random
# python terminal interaction library, not used for now
import argparse
# deque for replay buffer
import collections
# game, NNet and MCTS
from lib import model_pulse as model, mcts_pulse as mcts
import numpy as np
# summary show
from tensorboardX import SummaryWriter
# import hyperparameters
from lib.args import args
import torch
import torch.optim as optim
import torch.nn.functional as F


def evaluate(net1, net2, rounds, device="cpu",
             reward_threshold=0):  # competition game with apprentice and current best player
    '''
    Executes competitive playing between the Apprentice and the Best Player

    :param reward_threshold: current reward threshold for a state to be passed into the buffer
    :param net1: Apprentice net
    :param net2: Best Player net
    :param rounds: number of evaluation rounds
    :param device: cuda/cpu
    :return: win ratio
    '''
    n1_win, n2_win = 0, 0
    # initialize empty MCTS instances since we want to look at NNet's performance, not plain MCTS
    mcts_stores_double = [mcts.MCTS(), mcts.MCTS()]
    # we store Apprentice best reward for better observation of training process
    current_player_reward = []
    logs.info(f'Evaluation started')
    for r_idx in range(rounds):
        # both nets play with no exploration - they have only that they learned
        logs.warning(f'Round {r_idx + 1} started')
        r1, *_ = model.play_game(mcts_stores=mcts_stores_double[0], replay_buffer=None, net=net1,
                                 steps_before_tau_0=0, mcts_searches=config_training['MCTS_batch_searches'],
                                 mcts_batch_size=config_training['MCTS_batch_size'],
                                 device=device, reward_threshold=reward_threshold, enable_highlight=False)
        # logs.debug(f'R1:{r1:.3f}')
        current_player_reward.append(r1)
        r2, *_ = model.play_game(mcts_stores=mcts_stores_double[1], replay_buffer=None, net=net2,
                                 steps_before_tau_0=0, mcts_searches=config_training['MCTS_batch_searches'],
                                 mcts_batch_size=config_training['MCTS_batch_size'],
                                 device=device, reward_threshold=reward_threshold, enable_highlight=False)
        # logs.debug(f'R2:{r2:.3f}')
        if r1 < r2:
            n2_win += 1
            logs.error(f'Best Player has won')
        elif r1 > r2:
            n1_win += 1
            logs.error(f'Apprentice has won')
    if n1_win == n2_win == 0:
        logs.critical('No results during evaluation, keep the current net')
        return 0, max(current_player_reward)
    return n1_win / (n1_win + n2_win), max(current_player_reward)


if __name__ == "__main__":
    # console input not used for now
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    # args = parser.parse_args()

    # initial parameters
    config_training = args['training_config']
    config_eval = args['evaluation_config']
    min_replays = config_training['min_replay_to_train']
    device = torch.device("cuda" if args['environment'] == 'cuda' else "cpu")
    action_num = mcts.action_num
    replay_buffer = collections.deque(maxlen=config_training['replay_buffer'])
    mcts_store = mcts.MCTS()
    step_idx = 0
    best_idx = 0
    reward_threshold = args['reward_threshold']

    # saves path for models/run parameters (args)/best pulse lists
    saves_path = os.path.join(args['save_folder_name'], args['run_name'])
    os.makedirs(saves_path, exist_ok=True)
    args_saves_path = os.path.join(saves_path, 'args.txt')
    with open(args_saves_path, 'w+') as file:
        file.write(str(args))

    # Tensorboard
    writer = SummaryWriter(comment="-" + args['run_name'])

    # inform the user about the start
    logs.info(f'Filling the replay buffer, needed length is {min_replays}')

    # initialize NNets for Best Player and Apprentice
    net = model.Net(input_shape=model.observation_shape, actions_n=action_num).to(device)
    best_net = ptan.agent.TargetNet(net)
    optimizer = optim.SGD(net.parameters(), lr=config_training['learning_rate'],
                          momentum=config_training['SGD_momentum'])

    # show current config of Apprentice net
    print(net)

    # we track the net progress as it learns
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        # NNet trains until user stops the process
        while True:
            t = time.time()
            prev_nodes = len(mcts_store)
            game_steps = 0
            # play game, update reward threshold if necessary
            _, steps, new_threshold = model.play_game(mcts_store, replay_buffer, net=best_net.target_model,
                                                      steps_before_tau_0=config_training['steps_before_tau_0'],
                                                      mcts_searches=config_training['MCTS_batch_searches'],
                                                      mcts_batch_size=config_training['MCTS_batch_size'], device=device,
                                                      reward_threshold=reward_threshold, enable_highlight=False)
            game_steps += steps
            reward_threshold = new_threshold

            game_nodes = len(mcts_store) - prev_nodes
            # training speed characteristics
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt

            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            # training info is shown to a user
            logs.info(f"Step \u001b[32;1m%d\u001b[0m \
            leaves \u001b[37;1m%4d\u001b[0m \
            steps/s \u001b[37;1m%5.2f\u001b[0m \
            leaves/s \u001b[37;1m%6.2f\u001b[0m \
            win_eval_count \u001b[31;1m%d\u001b[0m \
            num_replays \u001b[37;1m%d\u001b[0m" % (
                step_idx, game_nodes, speed_steps, speed_nodes, best_idx, len(replay_buffer)))
            step_idx += 1

            # self-play games fill the replay buffer
            if len(replay_buffer) < config_training['min_replay_to_train']:
                continue

            # after the buffer is full enough, begin training
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(config_training['training_rounds']):
                # we sample a batch of random states from the buffer
                batch = random.sample(replay_buffer, config_training['training_batch_size'])
                batch_states, batch_probs, batch_values = zip(*batch)
                batch_states_lists = [state for state in batch_states]

                # and pass them to the NNet (Apprentice net)
                states_v = model.state_lists_to_batch(state_lists=batch_states_lists, device=device)
                optimizer.zero_grad()
                probs_v = torch.FloatTensor(batch_probs).to(device)
                values_v = torch.FloatTensor(batch_values).to(device)
                out_logits_v, out_values_v = net(states_v)

                # then compare the prediction of NNet with the real outcomes and calculate losses
                loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
                loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
                loss_policy_v = loss_policy_v.sum(dim=1).mean()

                loss_v = loss_policy_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                sum_loss += loss_v.item()
                sum_value_loss += loss_value_v.item()
                sum_policy_loss += loss_policy_v.item()

            tb_tracker.track("loss_total", sum_loss / config_training['training_rounds'], step_idx)
            tb_tracker.track("loss_value", sum_value_loss / config_training['training_rounds'], step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / config_training['training_rounds'], step_idx)

            # after certain amount of self-play games, Net_vs_Net evaluation games are performed
            if step_idx % config_eval['num_steps_before_evaluation'] == 0:
                win_ratio, best_eval_reward = evaluate(net, best_net.target_model,
                                                       rounds=config_eval['evaluation_rounds'],
                                                       device=device, reward_threshold=reward_threshold)
                logs.critical("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("Apprentice/Best Player win ratio", win_ratio, step_idx)
                writer.add_scalar("Best Apprentice evaluation reward", best_eval_reward, step_idx)
                # if Apprentice net has won the current Best Player, Apprentice becomes a new Best Player
                if win_ratio > config_eval['best_net_win_ratio']:  # >=
                    logs.critical("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(saves_path, "best_%03d_net_step_%05d.dat" % (best_idx, step_idx))
                    torch.save(net.state_dict(), file_name)
                    # MCTS buffers are cleared!
                    mcts_store.clear()
                    if args['reset_reward_threshold_after_eval']:
                        reward_threshold = args['reward_threshold']
