import math
import numpy as np
from lib.args import args
from numba import njit

'''
WARNING!
The following algorithm computes fidelity roughly for the sake of calculation speed!
'''
# parameters list
config = args['qbit_simulation_config']
pulse_time = config['pulse_time']
dimensions = config['n_dimensions']
omega_01 = config['omega_01']
mu = config['mu']
pulse_period = 2 * np.pi / config['omega_osc']
counts = config['num_timesteps']
mu = config['mu']
omega_01 = config['omega_01']
amp = config['amp']
tau = pulse_period / counts
pulse_period = 2 * np.pi / config['omega_osc']
counts = config['num_timesteps']


class Operator:
    def __init__(self, dimension, conjugate):
        self.dimension = dimension
        self.conjugate = conjugate
        self.matrix = np.zeros((self.dimension, self.dimension), dtype='complex128')
        try:
            n = self.dimension
            assert self.conjugate == 0 or self.conjugate == 1 == True
            if self.conjugate == 0:
                for i in range(n):
                    for j in range(n):
                        if i + 1 == j:
                            self.matrix[i][j] = np.sqrt(j)
            else:
                for i in range(n):
                    for j in range(n):
                        if j + 1 == i:
                            self.matrix[i][j] = np.sqrt(i)
        except ValueError:
            print('Please enter an integer!')
            self.matrix = None
        except AssertionError:
            print('PLease enter 0 for annihilation and 1 for creation')
            self.matrix = None

    def __repr__(self):
        return f'{self.matrix}'


class Psi:
    def __init__(self, dims):
        self.dimensions = dims
        self.matrix = np.zeros((self.dimensions, 1), dtype='complex128')
        self.matrix[0][0] = 1

    def __repr__(self):
        return f'{self.matrix}'


class PulsePreset:
    def __init__(self, pulse: int):
        self.pulse = pulse
        if self.pulse != 0:  # non-zero pulse
            self.oscillatory_part_pulse = amp * self.pulse
            self.pulse_counts = int(pulse_time / pulse_period * counts)

            self.pulse_part = self._calculate(self.oscillatory_part_pulse, self.pulse_counts)

            self.oscillatory_part_empty = 0
            self.empty_counts = int(counts - self.pulse_counts)

            self.empty_part = self._calculate(self.oscillatory_part_empty, self.empty_counts)

            self.total = self.pulse_part @ self.empty_part

        else:  # zero pulse
            self.oscillatory_part = 0
            self.empty_counts = int(counts)

            self.total = self._calculate(self.oscillatory_part, self.empty_counts)

    def __repr__(self):
        return f'{self.total}'

    @staticmethod
    def _calculate(_oscillatory_part: int, _counts: int):
        hamiltonian = omega_01 * creation @ annihilation - mu / 2 * creation @ annihilation * (
                creation @ annihilation - I) + _oscillatory_part * (creation + annihilation)

        F = hamiltonian

        numerator = I - (tau ** 2 / 12) * F @ F - 1j * (tau / 2) * F
        denominator = I - (tau ** 2 / 12) * F @ F + 1j * (tau / 2) * F
        denominator = np.linalg.inv(denominator)
        fraction = denominator @ numerator
        return np.linalg.matrix_power(fraction, _counts)


# identity matrix
I = np.identity(dimensions)
# operators of annihilation/creation
annihilation = Operator(dimensions, 0).matrix
creation = Operator(dimensions, 1).matrix
# initial psi matrix
psi_matrix = Psi(dimensions).matrix
# initial hamiltonian (for reliable calculation of eigenpsi)
hamiltonian_0 = omega_01 * creation @ annihilation - mu / 2 * creation @ annihilation * (
        creation @ annihilation - I)
eigenenergy, eigenpsi = np.linalg.eig(hamiltonian_0)
# initialize ground/excited/third state
ground_state, excited_state, third_state = eigenpsi[0].transpose(), eigenpsi[1].transpose(), eigenpsi[2].transpose()
# initialize precalculated matrices for positive/negative/zero pulse
positive, negative, empty = PulsePreset(1).total, PulsePreset(-1).total, PulsePreset(0).total

# desired_leakage = config['desired_leakage'] # deprecated?
'''
The process of reward calculation consists of several steps:
1) Apply precalculated pulse matrix stepwise for every pulse in a pulse list
2) Calculate probabilities for ground,excited and third state
3) Calculate fidelity in form of 1 - (p_groud + p_excited) - p_third
4) Return the characteristic proportional for fidelity but suitable for network rewards
'''


@njit(cache=True, fastmath=True, nogil=True)
def reward_calculation(
        pulse_list
):
    psi = psi_matrix
    for pulse in pulse_list:
        if pulse == 1:
            psi = positive @ psi
        elif pulse == -1:
            psi = negative @ psi
        elif pulse == 0:
            psi = empty @ psi

    psi_conj = psi.conjugate()
    probability_excited = excited_state @ psi_conj
    probability_ground = ground_state @ psi_conj
    probability_third = third_state @ psi_conj
    probability_excited = abs(np.sum(probability_excited)) ** 2
    probability_ground = abs(np.sum(probability_ground)) ** 2
    probability_third = abs(np.sum(probability_third)) ** 2
    fidelity = 1 - (probability_ground + probability_excited) - probability_third
    return fidelity


if __name__ == '__main__':
    # pulse list example made with genetic algorithm
    pulse_str = '-1-11111-1-1-1111111-1-101111-1-1-101111-1-1-1-11111-1-1-1-111110-1-1-101111-1-1-1-11111-1-1-1-111110\
    -1-1-101111-1-1-1-111111-1-1-111100-1-1-10111-1-1-1-1011-1-1-1-1-11111'
    # pulse list made by neural network
    pulse_str = ' -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1-1  -1  -1   1   1   1  -1  -1 \
    -1  -1  -1   1   1   1   1  -1  -1  -1-1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -11   1  \
     1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -1  -1   11   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1\
      -1   1   1   11  -1  -1  -1  -1   0   1   1   1   0  -1  -1  -1  -1   1   1   1   1-1  -1  -1   1   1   1   1   0\
        -1  -1  -1  -1'

    # from string to pulse list
    pulse_str = pulse_str.replace('1', '1,')
    pulse_str = pulse_str.replace('0', '0,')
    pulse_list = pulse_str.split(',')
    pulse_list.pop(-1)
    pulse_list = [int(pulse) for pulse in pulse_list]

    # speed demonstration
    # for _ in range(30000):
    #     reward_calculation(pulse_list=pulse_list)

    print(reward_calculation(pulse_list=pulse_list))
