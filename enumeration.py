from simulation_speed import reward_calculation
from lib.args import args

max_length = args['pulse_array_length']

pulse_list = [0]
for _ in range(max_length):
    scores_list = None
    pulse_list[-1] = 0
    zero = reward_calculation(pulse_list)
    pulse_list[-1] = 1
    positive = reward_calculation(pulse_list)
    pulse_list[-1] = -1
    negative = reward_calculation(pulse_list)
    if scores_list is None:
        scores_list = [negative,zero,positive]
    best_result = max(zero,positive,negative)
    pulse_value = scores_list.index(best_result) - 1
    pulse_list[-1] = pulse_value
    pulse_list.append(0)
    print(best_result)

trunc = 0
reward = 0
for i in range(1, 30):
    temp_reward = reward_calculation(pulse_list[:-i])
    if temp_reward > reward:
        trunc = i
        reward = temp_reward

pulse_list = pulse_list[:-trunc]

print(f'Fidelity: {reward_calculation(pulse_list):.3f}')
print(f'Pulse list:{pulse_list}')