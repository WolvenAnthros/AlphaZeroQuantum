import numpy as np
from numpy import complex128
from lib.args import args
from numpy.linalg import inv
from numba import njit
from scipy import optimize
import matplotlib.pyplot as plt

config = args['qbit_simulation_config']

# step 2: list of constants
dimensions = config['n_dimensions']
pulse_time = config['pulse_time']
theta = config['theta']
omega_01 = config['omega_01']
mu = omega_01 - config['mu']
pulse_period = 2 * np.pi / config['omega_osc']

tstep = config['num_timesteps']  # 1 /

h = 1.054e-34  # Planck constant
c1 = 1e-12  # Qubit capacity
f0 = 2.06e-15  # magnetic flux quantum
identity_matrix = np.identity(dimensions, dtype=np.complex128)  # identity
v = f0 / pulse_time  # voltage
c_c = theta / (f0 * np.sqrt(2 * omega_01 / (h * c1)))  # connection capacity
amp = c_c * v * np.sqrt(h * omega_01 / (2 * c1))  # pulse amp
print(f'Pulse amplitude: {amp:.3e}')
# hamiltonians
H0 = np.array([[0, 0, 0], [0, h * omega_01, 0], [0, 0, h * omega_01 + h * mu]], dtype=np.complex128)
eigenenergy, eigenpsi = np.linalg.eig(H0)
wave_func_0, wave_func_1, wave_func_2 = eigenpsi[:, [0]], eigenpsi[:, [1]], eigenpsi[:, [2]]
# ideal gate matrix
Y = 1 / np.sqrt(2) * (wave_func_1 @ wave_func_1.conj().T + wave_func_0 @ wave_func_0.conj().T
                      + wave_func_1 @ wave_func_0.conj().T - wave_func_0 @ wave_func_1.conj().T)


class U:
    Hmatrix = np.array([[0, -1, 0], [1, 0, -np.sqrt(2)], [0, np.sqrt(2), 0]], dtype=complex128)
    zero_enumerator = (identity_matrix - 1j * H0 * tstep / (2 * h))
    zero_denominator = inv((identity_matrix + 1j * H0 * tstep / (2 * h)))
    zero_pulse = zero_enumerator @ zero_denominator
    pulse_counts = int(pulse_time / tstep)
    empty_counts = int((pulse_period - pulse_time) / tstep)
    print(f'Pulse counts: {pulse_counts}, empty counts: {empty_counts}')

    def __init__(self, pulse: int):
        Hr_pulse = pulse * 1j * amp * U.Hmatrix + H0
        enumerator = (identity_matrix - 1j * Hr_pulse * tstep / (2 * h))
        denominator = inv((identity_matrix + 1j * Hr_pulse * tstep / (2 * h)))
        u_step = enumerator @ denominator
        u_pulse = u_t = identity_matrix
        for _ in range(U.pulse_counts):
            u_pulse = u_step @ u_pulse
        for _ in range(U.empty_counts):
            u_t = U.zero_pulse @ u_t
        self.matrix = u_t @ u_pulse


class AlphaState:
    def __init__(self, dim: int, n: int):
        assert n in range(1, 7, 1)
        alpha_1 = np.zeros((dim, 1), dtype='complex128')
        alpha_1[0] = 1
        alpha_2 = np.zeros((dim, 1), dtype='complex128')
        alpha_2[1] = 1
        if n == 1:
            self.matrix = alpha_1
        elif n == 2:
            self.matrix = alpha_2
        elif n == 3:
            self.matrix = 1 / np.sqrt(2) * (alpha_1 + alpha_2)
        elif n == 4:
            self.matrix = 1 / np.sqrt(2) * (alpha_1 - alpha_2)
        elif n == 5:
            self.matrix = 1 / np.sqrt(2) * (alpha_1 + alpha_2 * 1j)
        elif n == 6:
            self.matrix = 1 / np.sqrt(2) * (alpha_1 - alpha_2 * 1j)
        else:
            self.matrix = None


alpha_state_list = np.array([AlphaState(dimensions, i).matrix for i in range(1, 7)])  # initial conditions
u_t_plus, u_t_minus, u_t_zero = U(1).matrix, U(-1).matrix, U(0).matrix


@njit(fastmath=True)
def reward_calculation(pulse_list):
    fidelity = 0
    u_matrix = identity_matrix
    for pulse in pulse_list:
        if pulse == 1:
            u_matrix = u_t_plus @ u_matrix
        elif pulse == -1:
            u_matrix = u_t_minus @ u_matrix
        elif pulse == 0:
            u_matrix = u_t_zero @ u_matrix
    for alpha_state in alpha_state_list:
        ket = u_matrix @ alpha_state
        r_ket = Y @ alpha_state
        inner = r_ket.conj().T @ ket
        fidelity = fidelity + np.abs(inner[0][0]) ** 2
    return fidelity / 6


def umatrix_calculation(pulse_list):
    u_matrix = identity_matrix
    for pulse in pulse_list:
        if pulse == 1:
            u_matrix = u_t_plus @ u_matrix
        elif pulse == -1:
            u_matrix = u_t_minus @ u_matrix
        elif pulse == 0:
            u_matrix = u_t_zero @ u_matrix
    return u_matrix


def wait_calculation(num_timesteps, _u_matrix, ):
    matrices_list = []
    for steps in num_timesteps:
        _u_matrix = _u_matrix
        for _ in [steps]:
            _u_matrix = U.zero_pulse @ _u_matrix
        matrices_list.append(_u_matrix)
    fidelities = []
    for matrix in matrices_list:
        fidelity = 0
        for alpha_state in alpha_state_list:
            ket = matrix @ alpha_state
            r_ket = Y @ alpha_state
            inner = r_ket.conj().T @ ket
            fidelity = fidelity + np.abs(inner[0][0]) ** 2
        fidelity = fidelity / 6
        fidelity = 1 - fidelity  # REMIND: its infidelity!
        fidelities.append(fidelity)
    return fidelities


# # method 2
# fid_1 = 0
# for alpha_state in alpha_state_list:
#     ro = alpha_state @ alpha_state.conj().T
#     inner1 = umatrix @ ro @ umatrix.conj().T
#     inner2 = Y @ ro @ Y.conj().T
#     fid_1 = fid_1 + np.trace(inner1 @ inner2)
#     # print(abs(np.trace(inner1 @ inner2)))
# fid_1 = abs(fid_1) / 6

if __name__ == "__main__":
    pulse_str = config['example_scallop']
    pulse_str = pulse_str.replace('1', '1,')
    pulse_str = pulse_str.replace('0', '0,')
    pulse_list = pulse_str.split(',')
    pulse_list.pop(-1)
    pulse_list = [int(pulse) for pulse in pulse_list]

    u_matrix = umatrix_calculation(pulse_list)

    time_list = np.linspace(1, int(1e2), int(1e5), dtype=int)

    # time_tick = time_list[np.argmax(wait_calculation(_u_matrix=u_matrix, num_timesteps=time_list))]

    time_tick = optimize.fmin(func=wait_calculation, x0=[10], args=(u_matrix,))
    print(
        f'time ticks needed: {time_tick}, fidelity: {wait_calculation(_u_matrix=u_matrix, num_timesteps=[time_tick])}')
    plt.plot(time_list * config['num_timesteps'] * 1e6, wait_calculation(time_list, u_matrix))
    plt.show()
    print(Y)
    # trunc = 0
    # for i in range(1, 5):
    #     temp_reward = reward_calculation(pulse_list[:-i])
    #     if temp_reward > reward:
    #         trunc = i
    #         reward = temp_reward
    # print(f'Fidelity: {reward:.3g}, Infidelity: {1 - reward:.2e}, truncate {trunc} symbols for the best result')
