import math
import numpy as np
import matplotlib.pyplot as plt
from lib.args import args
import re


class Operator:
    def __init__(self, dimension, conjugate):
        self.dimension = dimension
        self.conjugate = conjugate
        self.matrix = np.zeros((self.dimension, self.dimension))
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
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.matrix = np.zeros((self.dimensions, 1))
        self.matrix[0][0] = 1


'''
Initial params
'''
config = args['qbit_simulation_config']


def simulation(
        pulse_list=None,
        dimension=config['n_dimensions'],
        omega_01=config['omega_01'],
        omega_osc = config['omega_osc'],
        amp=config['amp'],
        pulse_time=config['pulse_time'],
        mu=config['mu'],
):

    if pulse_list is None:
        pulse_list = [0, 0, 0]

    I = np.identity(dimension)  # Identity matrix
    annihilation = Operator(dimension, 0).matrix  # annihilation/creation operators
    creation = Operator(dimension, 1).matrix
    psi = Psi(dimension).matrix  # create psi matrix

    # Finding eigenvalues and eigenvectors
    Hamiltonian_0 = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
            np.matmul(creation, annihilation) - I)
    eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)

    # initial excited/ground state
    excited_state = np.array(eigenpsi[:, [1]])
    ground_state = np.array(eigenpsi[:, [0]])

    # end_probability_excited|ground|third
    for n in ['excited', 'ground', 'third']:
        globals()['end_probability_%s' % n] = []
    pulse_period = 2 * np.pi / omega_osc
    counts = int(pulse_period * config['num_timesteps'] )  # let's take a lot of counts for more reliability

    #tau = pulse_period / counts  # 0.0004
    tau = pulse_period/counts
    for pulse, index in zip(pulse_list, range(len(pulse_list))):
        '''
        Please pay attention that the pulse_period and pulse_time should converge well, otherwise
        we lose some of the counts
        '''
        # print(f'Pulse:{pulse}, index: {index}')
        for k in range(counts):
            # Start of psi calculation
            t = lambda k_: k_ * tau
            '''
            Defining pulse shape in time
            '''
            if t(k) < pulse_time:
                oscillatory_part = amp * pulse
                oscillatory_part_1st_derivative = 0
                oscillatory_part_2nd_derivative = 0
            else:
                oscillatory_part = 0
                oscillatory_part_1st_derivative = 0
                oscillatory_part_2nd_derivative = 0

            # Crank-Nicolson 2nd order method implementation

            Hamiltonian = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
                    np.matmul(creation, annihilation) - I) + oscillatory_part * (creation + annihilation)
            Hamiltonian_1st_derivative = oscillatory_part_1st_derivative
            Hamiltonian_2nd_derivative = oscillatory_part_2nd_derivative
            commutator = Hamiltonian_1st_derivative * Hamiltonian - Hamiltonian * Hamiltonian_1st_derivative

            F = Hamiltonian + (tau ** 2 / 24) * Hamiltonian_2nd_derivative - 1j * (tau ** 2 / 12) * commutator

            numerator = I - (tau ** 2 / 12) * np.matmul(F, F) - 1j * (tau / 2) * F
            denominator = I - (tau ** 2 / 12) * np.matmul(F, F) + 1j * (tau / 2) * F
            denominator = np.linalg.inv(denominator)
            fraction = denominator @ numerator

            psi = fraction @ psi

            # excited probability formula
            probability_excited = excited_state.transpose() @ psi.conjugate()
            probability_excited = abs(np.sum(probability_excited)) ** 2
            end_probability_excited.append(probability_excited)

            # ground probability formula
            probability_ground = ground_state.transpose() @ psi.conjugate()
            probability_ground = abs(np.sum(probability_ground)) ** 2
            end_probability_ground.append(probability_ground)

            if dimension >= 3:
                third_state = np.array(eigenpsi[:, [2]])

                # third state probability formula
                probability_third = third_state.transpose() @ psi.conjugate()
                probability_third = abs(np.sum(probability_third)) ** 2
                end_probability_third.append(probability_third)
                if abs(probability_excited - 0.5) < 0.015:
                    print(f'time: {t(k)}, count: {k}')
                    leakage = 0
                    for dim in range(2, dimension):
                        high_state = np.array(eigenpsi[:, [dim]])
                        # third state probability formula
                        probability_high = high_state.transpose() @ psi.conjugate()
                        probability_high = abs(np.sum(probability_high)) ** 2
                        leakage += probability_high
                    print(f'leakage: {leakage}')
                    print('\n')
                    return 1, end_probability_ground, end_probability_excited, end_probability_third

            else:
                probability_third = 0
                end_probability_third.append(probability_third)

    return abs(probability_excited * 2), end_probability_ground, end_probability_excited, end_probability_third


def plot_show(ground,
              excited,
              third,
              omega_osc=25 * 2 * np.pi):
    pulse_period = 2 * np.pi / omega_osc
    counts = int(pulse_period * 2500)
    tau = pulse_period / counts  # 0.0004

    axis = [x * tau for x in range(len(excited))]
    fig, at = plt.subplots()
    at.plot(axis, excited, label='excited state')
    at.plot(axis, ground, label='ground state')
    at.plot(axis, third, label='third state')
    at.set_xlabel('time, ns')
    at.set_ylabel('probability')
    at.set_title("Leakage graph")
    # at.legend(loc='lower left')
    # plt.yscale("log")
    plt.show()


num_timestemps = config['num_timesteps']


def pulse_show(pulse_list):
    pulse_period = 2 * np.pi / config['omega_osc']
    tau = pulse_period / num_timestemps
    axis = [x * tau for x in range(len(pulse_list) * num_timestemps)]

    pulses = []
    for pulse in pulse_list:
        pulse_timed = [pulse * config['amp'] if x < config['pulse_time'] else 0 for x in range(num_timestemps)]
        pulses += pulse_timed
    fig, at = plt.subplots()
    at.plot(axis, pulses, label='pulses')
    at.set_xlabel('time, ns')
    at.set_ylabel('amplitude')
    plt.show()


if __name__ == '__main__':

    pulse_str = config['example_scallop']
    pulse_str = pulse_str.replace('1', '1,')
    pulse_str = pulse_str.replace('0', '0,')
    pulse_list = pulse_str.split(',')
    pulse_list.pop(-1)
    pulses = re.findall(r'[+-]?\d', pulse_str)
    pulses_str = ''

    for pulse in pulses:
        pulses_str += pulse

    pulse_list = [int(pulse) for pulse in pulse_list]
    print(pulses_str)
    win, end_probability_ground, end_probability_excited, end_probability_third = simulation(pulse_list=pulse_list)
    print(f'Reward: {win:.3e}')
    plot_show(ground=end_probability_ground, excited=end_probability_excited, third=end_probability_third)
    # pulse_show(pulse_list)
