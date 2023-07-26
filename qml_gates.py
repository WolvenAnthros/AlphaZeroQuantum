from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit import BasicAer
import numpy as np
from lib.args import args
from itertools import permutations

backend = BasicAer.get_backend('statevector_simulator')
num_qubits = 5  # число кубитов в системе

vec = np.array([0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                4.32978028e-17 + 0.70710678j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                4.32978028e-17 + 0.70710678j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j,
                0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j, ])
target_state = Statevector(vec, dims=(2, 2, 2, 2, 2))

print(target_state.to_dict())

INITIAL_STATE = []
INITIAL_STATE = tuple(INITIAL_STATE)
INITIAL_INDEX = 0
game_length = args['pulse_array_length']
qubit_indices = [x for x in range(num_qubits)]
# TODO: optimize code
max_single_qubit_permutations = num_qubits
max_two_qubit_permutations = 20
max_three_qubit_permutations = 60
two_qubit_perm, three_qubit_perm = [], []
perm_two = permutations(qubit_indices, 2)
for val in perm_two:
    two_qubit_perm.append(val)

perm_three = permutations(qubit_indices, 3)
for val in perm_three:
    three_qubit_perm.append(val)


def move(state, idx, action):
    """

    :param state: квантовое состояние
    :param idx: номер операции
    :param action: вид операции
    :return:
    """
    state = list(state)  # decode

    q = QuantumRegister(num_qubits)

    qc = QuantumCircuit(q)

    singe_qubit_operations = [qc.id, qc.x, qc.y, qc.z, qc.h, qc.s, qc.sdg, qc.t]
    two_qubit_operations = [qc.cx, qc.cy, qc.cz, qc.ch, qc.swap]
    three_qubit_operations = [qc.ccx, qc.cswap]

    rotational_gates = [qc.u, qc.p, qc.rx, qc.ry, qc.rz,
                        qc.crz, qc.cp, qc.cu]

    two_qubit_threshold = max_two_qubit_permutations * len(two_qubit_operations) + \
                          max_single_qubit_permutations * len(singe_qubit_operations)

    state.append(action)
    for action in state:
        if action < max_single_qubit_permutations * len(singe_qubit_operations):  # single qubit operation
            action_type = action // num_qubits
            qubit_index = action % num_qubits

            action_type = singe_qubit_operations[action_type]  # quantum gate
            action_type(qubit_index)

        elif two_qubit_threshold > \
                action >= max_single_qubit_permutations * len(singe_qubit_operations):  # two-qubit operation
            action = action - num_qubits * 8
            max_qubit_combinations = num_qubits * 4  # FIXME: check formula
            action_type = action // max_qubit_combinations  # REMIND: check this formula for different number of qubits
            action = action - action_type * max_qubit_combinations
            first_qubit_index, second_qubit_index = two_qubit_perm[action]

            action_type = two_qubit_operations[action_type]
            action_type(first_qubit_index, second_qubit_index)

        elif action >= two_qubit_threshold:
            action = action - two_qubit_threshold
            max_qubit_combinations = num_qubits * 24  # REMIND: check this formula for different number of qubits
            action_type = action // max_qubit_combinations
            action = action - action_type
            first_qubit_index, second_qubit_index, third_qubit_index = three_qubit_perm[action]

            action_type = three_qubit_operations[action_type]
            action_type(first_qubit_index, second_qubit_index, third_qubit_index)

    job = backend.run(transpile(qc, backend))
    qc_state = job.result().get_statevector(qc)
    reward = state_fidelity(target_state, qc_state)
    print(qc.draw(output='text'))
    print(f'Infidelity: {1 - reward:.2e}')

    done = False
    if idx == game_length - 2:
        done = True

    state = tuple(state)
    return state, reward, done


if __name__ == "__main__":
    state, reward, done = move(INITIAL_STATE, 0, 13)
    state, reward, done = move(state, 1, 24)
    state, reward, done = move(state, 2, 140)  # 140 - first three-qubit operation
    state, reward, done = move(state, 2, 138)
    state, reward, done = move(state, 2, 5)
    print(f'Gates list: {state}')
