import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from lib.args import args
import re

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


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
# phase = 0

# pulses = [1 for x in range(120)]


# Irregular pulse array example
# pulse_list = [-1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,-1,0,1,1,1,1,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,0,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,0,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,0,0,-1,-1,-1,0,1,1,1,-1,-1,-1,-1,0,1,1,-1,-1,-1,-1,-1,1,1,1,1
# ]

pulse_list = [0,1,1,1,-1,0,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,-1,-1,0,1,1,1,-1,0,1,1,-1,-1,0,1,1,-1,-1,1,1,-1,-1,-1,0,1,1,-1,1,0,1,-1,-1,-1,0,1,-1,-1,-1]

# pulse_str ='-1-11111-1-1-1111111-1-101111-1-1-101111-1-1-1-11111-1-1-1-111110-1-1-101111-1-1-1-11111-1-1-1-111110-1-1-101111-1-1-1-111111-1-1-111100-1-1-10111-1-1-1-1011-1-1-1-1-11111'

# pulse_str ='01111-1-1-1111-1-11111-1-10110-1-10111-1-11111-1-1-1111-1-1-11110-1-10111-1-1-1111-1-1-1111-1-1-11110-1-10111-1-1-1111-1-1-1111-1-1-11110-1-1011-1-1-11111-1-1111-1-1-111'

# pulse_str = '1011-1-11-1-111-1011-111-1111-111-101-1-111-1110-111-1-111-111-1-111-101-1-111-1-110-111-1-111-111-1-111-101-1-111-1-110-111-1-111-111-1-110-101-1-111-111-1-11-1-111-1-11'


config = args['qbit_simulation_config']

#pulse_list=[]
# pulse_show = load_object('pulse.pickle')
# print(pulse_show)
# print(pulse_list)

def simulation(
        pulse_list=config['default_pulse_list'],
        dimension=config['n_dimensions'],
        omega_01=config['omega_01'],
        omega_osc=config['omega_osc'],
        amp=config['amp'],
        pulse_time=config['pulse_time'],
        mu=config['mu'],
):
    pulse_period = 2 * np.pi / omega_osc

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

    for pulse, index in zip(pulse_list, range(len(pulse_list))):
        '''
        Please pay attention that the pulse_period and pulse_time should converge well, otherwise
        we lose some of the counts
        '''
        counts = int(pulse_period * 2500)  # let's take a lot of counts for more reliability

        tau = pulse_period / counts  # 0.0004
        #print(f'Pulse:{pulse}, index: {index}')
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
            fraction = np.matmul(denominator, numerator)

            psi = np.matmul(fraction, psi)

            # excited probability formula
            probability_excited = np.matmul(excited_state.transpose(), psi.conjugate())
            probability_excited = abs(np.sum(probability_excited)) ** 2
            end_probability_excited.append(probability_excited)

            # ground probability formula
            probability_ground = np.matmul(ground_state.transpose(), psi.conjugate())
            probability_ground = abs(np.sum(probability_ground)) ** 2
            end_probability_ground.append(probability_ground)

            if dimension >= 3:
                third_state = np.array(eigenpsi[:, [2]])

                # third state probability formula
                probability_third = np.matmul(third_state.transpose(), psi.conjugate())
                probability_third = abs(np.sum(probability_third)) ** 2
                end_probability_third.append(probability_third)
                # if index == (len(pulse_list)-1) and k == counts - 1:
                if abs(probability_excited - 0.5) < 0.015:
                    print(f'time: {t(k)}, count: {k}')
                    leakage = 0
                    for dim in range(2, dimension):
                        high_state = np.array(eigenpsi[:, [dim]])
                        # third state probability formula
                        probability_high = np.matmul(high_state.transpose(), psi.conjugate())
                        probability_high = abs(np.sum(probability_high)) ** 2
                        leakage += probability_high
                        # print('leakage:', leakage)
                    print(f'leakage: {leakage}')
                    print('\n')
                    return 1,end_probability_ground,end_probability_excited,end_probability_third

            else:
                third_state = 0
                probability_third = 0
                end_probability_third.append(probability_third)
                #PROBA *2 BECAUSE OF MODELLING ONLY HALF OF THE OPERATION
    #print(f'Excited state probability : {abs(probability_excited)}')
    return abs(probability_excited*2),end_probability_ground,end_probability_excited,end_probability_third



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
    at.legend(loc='lower left')
    # plt.yscale("log")
    plt.show()


if __name__ == '__main__':
    pulse_str = '-1-11111-1-1-1111111-1-101111-1-1-101111-1-1-1-11111-1-1-1-111110-1-1-101111-1-1-1-11111-1-1-1-111110-1-1-101111-1-1-1-111111-1-1-111100-1-1-10111-1-1-1-1011-1-1-1-1-11111'
    pulse_str =   '-1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1-1  -1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1-1   1   1   1   1  -1  -1  -1  -1   1   1   1   1   0  -1  -1  -1   01   1   1  -1  -1  -1  -1   0   1   1   1   0  -1  -1  -1   0   0   1 1   1   0  -1  -1  -1   0   1   1   1   0  -1  -1  -1   0   1   1   10   0  -1  -1   0   0   1   1  -1  -1  -1  -1  -1   0   1   1   1   0-1  -1  -1  -1   1   1   1   0   0  -1  -1  -1'

    pulse_str = ' -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1-1  -1  -1   1   1   1  -1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1-1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -11   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -1  -1   11   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -1   1   1   11  -1  -1  -1  -1   0   1   1   1   0  -1  -1  -1  -1   1   1   1   1-1  -1  -1   1   1   1   1   0  -1  -1  -1  -1'

    pulse_str = '-1  -1  -1  -1   1   1   1   0  -1  -1  -1  -1   1   1   1   0  -1  -1-1  -1   1   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1-1   1   1   1   0  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   01   1   1   0  -1  -1  -1   0   1   1   1   0  -1  -1  -1   0   1   11   0  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   0   1   1   10  -1  -1  -1   0   1   1   1   0  -1  -1  -1  -1   1   1   1   0  -1-1  -1  -1   0   1   1   1   0  -1  -1  -1   0'

    pulse_str =  '1  -1   1   1   1   1   1   1  -1  -1  -1   1   1   1   1   1  -1  -1-1   1   1   1   1   1  -1  -1  -1  -1   1   1   1   1   1  -1  -1  -1-1   1   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   11   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   1   11   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1  -1   1   1   11  -1  -1  -1   1   1   1   1   1  -1  -1  -1  -1   1   1   1  -1  -1-1  -1   1   1   1'

    pulse_str = '-1  -1  -1   1   1   1  -1  -1  -1  -1   1   1   1   1   1  -1-1  -1  -1   1   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1-1   1   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -11   1   1   1  -1  -1  -1  -1  -1   1   1   1  -1  -1  -1  -1  -1   11   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1  -1   1   11   1  -1  -1  -1   1   1   1   1   1  -1  -1  -1  -1   1   1   1   1-1  -1  -1  -1  -1   1   1      0   0   0   0'


    pulse_str = pulse_str.replace('1', '1,')
    pulse_str = pulse_str.replace('0', '0,')
    pulse_list = pulse_str.split(',')
    pulse_list.pop(-1)
    pulse = re.findall(r'[+-]?\d',pulse_str)
    pulse_list = [int(pulse) for pulse in pulse_list]
    win, end_probability_ground,end_probability_excited,end_probability_third = simulation(pulse_list=pulse_list)
    print(f'Reward: {win}')
    plot_show(ground=end_probability_ground, excited=end_probability_excited, third=end_probability_third)
