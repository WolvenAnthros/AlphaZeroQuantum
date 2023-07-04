import math
import numpy as np
import plotly.graph_objects as go
from logger import logger as logs
from simulation_speed import reward_calculation, wait_calculation
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
dimensions = config['n_dimensions']
omega_01 = config['omega_01']
omega_osc = config['omega_osc']
mu = config['mu']
pulse_time = config['pulse_time']
pulse_period = 2 * np.pi / omega_osc
counts = 10
tau = pulse_period / counts

I = np.identity(dimensions)  # Identity matrix
annihilation = Operator(dimensions, 0).matrix  # annihilation/creation operators
creation = Operator(dimensions, 1).matrix

# Finding eigenvalues and eigenvectors
Hamiltonian_0 = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
        np.matmul(creation, annihilation) - I)
eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)

# initial excited/ground state
excited_state = np.array(eigenpsi[:, [1]])
ground_state = np.array(eigenpsi[:, [0]])


def simulation(
        pulse_list=None,
        amp=4,
        pulse_time=config['pulse_time']
):
    if pulse_list is None:
        raise ValueError("No pulse list provided")

    psi = Psi(dimensions).matrix  # create psi matrix

    # end_probability_excited|ground|third
    for n in ['excited', 'ground', 'third']:
        globals()['end_probability_%s' % n] = []

    for pulse, index in zip(pulse_list, range(len(pulse_list))):

        # print(f'Pulse:{pulse}, index: {index}')
        for k in range(counts):

            # Start of psi calculation
            t = lambda k_: k_ * tau
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
            probability_excited = excited_state.T.conjugate() @ psi
            probability_excited = abs(np.sum(probability_excited)) ** 2
            end_probability_excited.append(probability_excited)

            # ground probability formula
            probability_ground = ground_state.T.conjugate() @ psi
            probability_ground = abs(np.sum(probability_ground)) ** 2
            end_probability_ground.append(probability_ground)

            if dimensions >= 3:
                third_state = np.array(eigenpsi[:, [2]])

                # third state probability formula
                probability_third = third_state.T.conjugate() @ psi
                probability_third = abs(np.sum(probability_third)) ** 2
                end_probability_third.append(probability_third)

                if abs(probability_excited - 0.5) < 0.025:
                    # print(f'time: {t(k)}, count: {k}')
                    leakage = 0
                    for dim in range(2, dimensions):
                        high_state = np.array(eigenpsi[:, [dim]])
                        # third state probability formula
                        probability_high = high_state.transpose() @ psi.conjugate()
                        probability_high = abs(np.sum(probability_high)) ** 2
                        leakage += probability_high


                    # return 1, end_probability_ground, end_probability_excited, end_probability_third

            else:
                probability_third = 0
                end_probability_third.append(probability_third)

    logs.info(f'leakage: {leakage:.2e}')

    return abs(probability_excited * 2), end_probability_ground, end_probability_excited, end_probability_third


def plot_show(ground,
              excited,
              third, fidelity_):
    axis = [x / len(excited) * 2 * np.pi / omega_osc * len(pulse_list) for x in range(len(excited))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=axis,
        y=ground,
        mode='lines',
        name='ground state'
    ))
    fig.add_trace(go.Scatter(
        x=axis,
        y=excited,
        mode='lines',
        name='excited state'
    ))
    fig.add_trace(go.Scatter(
        x=axis,
        y=third,
        mode='lines',
        name='leakage'
    ))
    leakage_annotation = [dict(
        x=axis[-1],
        y=third[-1],
        font=dict(size=16),
        text=f"Leakage: {third[-1]:.2e}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.8,
        ay=-80,
        ax=-100,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=1,
    )]
    infidelity_annotation = [dict(
        text=f"Infidelity: {1 - fidelity_:.2e}",
        font=dict(size=16),
        bordercolor="#c7c7c7",
        borderwidth=2,
        bgcolor="#ff7f0e",
        xref="paper", yref="paper",
        x=0.05, y=0.8,
        showarrow=False
    )]

    fig.update_layout(
        updatemenus=[
            dict(buttons=list([
                dict(label="None",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "States dynamics plot", "annotations": []}]),
                dict(label="Leakage",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "Leakage display", "annotations": leakage_annotation}]),
                dict(label="Infidelity",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "Infidelity display", "annotations": infidelity_annotation}]),
                dict(label="Leakage + Infidelity",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "Infidelity and leakage display",
                            "annotations": leakage_annotation + infidelity_annotation}]),
            ]),
            )
        ]
    )

    fig.update_layout(
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='rgb(0, 0, 0)',
            showgrid=True,
            showticklabels=True,
            ticks='outside'
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='rgb(0, 0, 0)',
            showgrid=True,
            showticklabels=True,
            ticks='outside'
        ),
        autosize=False,
        width=1000,
        height=700,
        plot_bgcolor='white'
    )
    fig.update_layout(title='State dynamic graph',
                      xaxis_title='Time, ns',
                      yaxis_title='Probability')
    plot_config = {'scrollZoom': True}
    fig.show(config=plot_config)


num_timestemps = config['num_timesteps']


def pulse_show(pulse_list):
    pulse_period = 2 * np.pi / config['omega_osc']

    axis = [x * pulse_period for x in range(len(pulse_list))]
    positive_pulse_list = [pulse if pulse > 0 else 0 for pulse in pulse_list]
    negative_pulse_list = [pulse if pulse < 0 else 0 for pulse in pulse_list]
    fig = go.Figure(data=
                    [go.Bar(x=axis, y=negative_pulse_list),
                     go.Bar(x=axis, y=positive_pulse_list),
                     ]
                    )

    plot_config = {'scrollZoom': True}
    fig.show(config=plot_config)


if __name__ == '__main__':

    pulse_str = config['example_scallop']  # leakage ~10^-4
    pulse_str = pulse_str.replace('1', '1,')
    pulse_str = pulse_str.replace('0', '0,')
    pulse_list = pulse_str.split(',')
    pulse_list.pop(-1)
    pulses = re.findall(r'[+-]?\d', pulse_str)
    pulses_str = ''

    for pulse in pulses:
        pulses_str += pulse

    pulse_list = [int(pulse) for pulse in pulse_list]
    print(pulse_list)

    _, end_probability_ground, end_probability_excited, end_probability_third = simulation(pulse_list=pulse_list)

    fidelity = reward_calculation(pulse_list=pulse_list)
    logs.debug(
        f'Fidelity:{fidelity:.4g}')
    logs.critical(
        f'Infidelity:{1 - fidelity:.3e}')
    logs.info(len(pulse_list))

    plot_show(ground=end_probability_ground, excited=end_probability_excited, third=end_probability_third,
              fidelity_=fidelity)
    pulse_show(pulse_list)
