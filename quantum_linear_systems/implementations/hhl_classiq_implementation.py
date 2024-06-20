"""HHL implementation using Classiq."""
import time
from itertools import product
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from classiq import execute
from classiq import GeneratedCircuit
from classiq import Model
from classiq import show
from classiq import synthesize
from classiq.builtin_functions import AmplitudeLoading
from classiq.builtin_functions import Exponentiation
from classiq.builtin_functions import PhaseEstimation
from classiq.builtin_functions import StatePreparation
from classiq.builtin_functions.exponentiation import ExponentiationOptimization
from classiq.builtin_functions.exponentiation import PauliOperator
from classiq.execution import ClassiqBackendPreferences
from classiq.execution import ExecutionPreferences
from classiq.interface.executor.quantum_program import QuantumProgram
from classiq.interface.generator.amplitude_loading import AmplitudeLoadingImplementation
from classiq.interface.generator.qpe import ExponentiationScaling
from classiq.interface.generator.qpe import ExponentiationSpecification
from classiq.synthesis import set_execution_preferences

from quantum_linear_systems.implementations.vqls_qiskit_implementation import (
    postprocess_solution,
)
from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample


Paulidict = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
}


# generate all combinations of Pauli strings of size n
def generate_all_pauli_strings(seq: str, size: int) -> Generator[str, None, None]:
    """Generate all combinations of Pauli strings of size n.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).
        size (int): The size of the output Pauli strings.

    Returns:
        A generator of all combinations of Pauli strings of size n.
    """
    for string in product(seq, repeat=size):
        yield "".join(string)


# convert a Paulistring of size n to 2**n X 2**n matrix
def pauli_string_2mat(seq: str) -> np.matrix:
    """Convert a Pauli string of size n to a 2**n x 2**n matrix.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).

    Returns:
        A 2**n x 2**n matrix representation of the Pauli string.
    """
    p_matrix = Paulidict[seq[0]]
    for p_string in seq[1:]:
        p_matrix = np.kron(p_matrix, Paulidict[p_string])
    return p_matrix


# Hilbert-Schmidt-Product of two matrices M1, M2
def hilbert_schmidt(m_1: np.ndarray, m_2: np.ndarray) -> Any:
    """Compute the Hilbert-Schmidt-Product of two matrices M1, M2.

    Parameters:
        m_1 (np.ndarray): The first matrix.
        m_2 (np.ndarray): The second matrix.

    Returns:
        The Hilbert-Schmidt inner product of the two matrices.
    """
    return (np.dot(m_1.conjugate().transpose(), m_2)).trace()


# Naive decomposition, running over all HS products for all Pauli strings
def lcu_naive(herm_mat: np.ndarray) -> List[Tuple[str, float]]:
    """Naive LCU (linear combination of unitary operations) decomposition, running over
    all HS products for all Pauli strings.

    Parameters:
        herm_mat (np.ndarray): The input Hermitian matrix.

    Returns:
        A list of tuples, each containing a Pauli string and the corresponding coefficient.
    """
    assert herm_mat.shape[0] == herm_mat.shape[1], "matrix is not square"
    assert herm_mat.shape[0] != 0, "matrix is of size 0"
    assert herm_mat.shape[0] & (herm_mat.shape[0] - 1) == 0, "matrix size is not 2**n"

    num_qubits = int(np.log2(herm_mat.shape[0]))
    pauli_strings = list(generate_all_pauli_strings("IZXY", num_qubits))

    mylist: List[Tuple[str, float]] = []

    for pstr in pauli_strings:
        coeff = (1 / 2**num_qubits) * hilbert_schmidt(pauli_string_2mat(pstr), herm_mat)
        if coeff != 0:
            mylist = mylist + [(pstr, coeff)]

    return mylist


def verify_matrix_sym_and_pos_ev(mat: np.ndarray) -> None:
    """Verify that the input matrix is symmetric and has positive eigenvalues.

    Parameters:
        mat (np.ndarray): The input matrix.
    """
    if not np.allclose(mat, mat.T, rtol=1e-6, atol=1e-6):
        raise ValueError("The matrix is not symmetric")
    eigenvalues, _ = np.linalg.eig(mat)
    for lam in eigenvalues:
        if lam < 0:
            raise ValueError("The matrix has negative eigenvalues")
        # the original classiq workshop checked if the EV are in [0,1)
        if lam > 1:
            print("The matrix has eigenvalues larger than 1: ", lam)


def state_preparation(vector_b: np.ndarray, sp_upper: float) -> StatePreparation:
    """Prepare the state based on the input vector.

    Parameters:
        vector_b (np.ndarray): The input vector.
        sp_upper (float): The upper bound of the L2 error metric for the state preparation.

    Returns:
        A StatePreparation object with the desired amplitudes and error metric.
    """
    vector_b = tuple(vector_b)
    # sp_upper = precision of the State Preparation
    return StatePreparation(
        amplitudes=vector_b, error_metric={"L2": {"upper_bound": sp_upper}}
    )


def quantum_phase_estimation(
    paulis: List[np.matrix],
    qpe_register_size: int,
    scaling: float,
    max_error: float,
    neg_vals: bool = True,
) -> PhaseEstimation:
    """Perform Quantum Phase Estimation (QPE) with the specified precision.

    Parameters:
        paulis (list) : List of pauli matrices.
        qpe_register_size (int): The desired size of the QPE register.

    Returns:
        A PhaseEstimation object configured with the specified precision.
    """
    exp_params = Exponentiation(
        pauli_operator=PauliOperator(pauli_list=paulis),
        evolution_coefficient=-2 * np.pi,
        # constraints=ExponentiationConstraints(max_error=max_error),
        optimization=ExponentiationOptimization.MINIMIZE_DEPTH,
    )
    return PhaseEstimation(
        size=qpe_register_size,
        unitary_params=exp_params,
        exponentiation_specification=ExponentiationSpecification(
            scaling=ExponentiationScaling(
                max_depth=250,
                # todo: why not precision
                max_depth_scaling_factor=1 + scaling,
            )
        ),
    )


def extract_solution(
    qprog_hhl: QuantumProgram,
    w_min: float,
    matrix_a: np.ndarray,
    vec_b: np.ndarray,
    smallest_ev: float,
    qpe_register_size: int,
) -> np.ndarray:
    """Extract the solution vector from the synthesized quantum program."""
    res_hhl = execute(qprog_hhl).result()[0].value

    total_q = GeneratedCircuit.from_qprog(
        qprog_hhl
    ).data.width  # total number of qubits of the whole circuit

    target_pos = res_hhl.physical_qubits_map["target"][0]  # position of control qubit

    sol_pos = list(res_hhl.physical_qubits_map["solution"])  # position of solution

    phase_pos = [
        total_q - k - 1 for k in range(total_q) if k not in sol_pos + [target_pos]
    ]  # finds the position of the “phase” register, and
    # flips for endianness as we will use the indices to read directly from the string
    qsol = [
        np.round(parsed_state.amplitude / w_min, 5)
        for solution in range(2 ** int(np.log2(matrix_a.shape[0])))
        for parsed_state in res_hhl.parsed_state_vector
        if (
            parsed_state["target"] == 1.0
            and parsed_state["solution"] == solution
            and
            # this takes the entries where the “phase” register is at state zero
            [parsed_state.bitstring[k] for k in phase_pos] == ["0"] * qpe_register_size
        )
    ]
    quantum_solution = np.array(qsol)
    print(
        "euclidian norm",
        np.linalg.norm(quantum_solution),
        np.linalg.norm(quantum_solution) / smallest_ev,
    )
    # print("state_vector", res_hhl.state_vector)

    global_phase = np.angle(quantum_solution)
    qsol_corrected = np.real(quantum_solution / np.exp(1j * global_phase))
    # normalize
    # todo: this is a hack to obtain the multiplication factor of the normalized quantum solution
    # qsol_corrected = normalize_quantum_by_classical_solution(qsol_corrected, sol_classical)
    qsol_corrected = postprocess_solution(
        matrix_a=matrix_a, vector_b=vec_b, solution_x=qsol_corrected
    )

    # Note: this is currently included in postprocess_solution
    # if vec_b_expanded:
    #     qsol_corrected = extract_x_from_expanded(qsol_corrected)

    return qsol_corrected


def classiq_hhl_implementation(
    matrix_a: np.ndarray, vector_b: np.ndarray, qpe_register_size: Optional[int] = None
) -> Tuple[QuantumProgram, np.ndarray, np.ndarray, float, int]:
    """Classiq HHL implementation based on
    https://docs.classiq.io/latest/tutorials/advanced/hhl/ ."""
    # verifying that the matrix is symmetric and hs eigenvalues in [0,1)
    # verify_matrix_sym_and_pos_ev(mat=matrix_a)

    epsilon = 1e-2
    epsilon_s = epsilon / 3  # state preparation
    # epsilon_r = epsilon / 3  # conditioned rotation
    epsilon_a = epsilon / 6  # hamiltonian simulation
    # Note: value of sp_upper: qiskit hhl.py line 101: epsilon=1e-2, line 120 state prep.: epsilon_s = epsilon / 3

    solution_register_size = int(np.log2(len(vector_b)))
    if qpe_register_size is None:
        # calculate size of qpe_register from matrix
        kappa = np.linalg.cond(matrix_a)  # condition number of matrix
        neg_vals = True  # whether matrix has negative eigenvalues
        qpe_register_size = (
            max(solution_register_size + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals
        )
    print(
        f"Size of solution register is {solution_register_size} , QPE registers is {qpe_register_size}."
    )

    model_hhl = Model()
    # Step 1: state preparation
    sp_out = model_hhl.StatePreparation(
        params=state_preparation(vector_b=vector_b, sp_upper=epsilon_s)
    )

    # Step 2 : Quantum Phase Estimation
    # lambda_max = max(np.abs(np.linalg.eigvals(matrix_a)))
    lambda_min = min(np.abs(np.linalg.eigvals(matrix_a)))
    exponentiation_scaling = lambda_min
    exponentiation_scaling = np.sqrt(2)
    print("classiq scaling is now", exponentiation_scaling)
    qpe = quantum_phase_estimation(
        paulis=lcu_naive(matrix_a),
        qpe_register_size=qpe_register_size,
        scaling=exponentiation_scaling,
        max_error=epsilon_a,
    )  # Todo: test if this explodes circuit depth
    qpe_out = model_hhl.PhaseEstimation(params=qpe, in_wires={"IN": sp_out["OUT"]})

    # Step 3 : Eigenvalue Inversion / Conditioned Rotation
    # Note: here qiskit seems to implement ExactReciprocal instead?! also investigate what scaling (delta) it uses
    w_min = (
        1 / 2**qpe_register_size
    )  # for qpe register of size m, this is the minimal value which can be encoded
    al_out = model_hhl.AmplitudeLoading(
        params=AmplitudeLoading(
            size=qpe_register_size,
            expression=f"{w_min}/(x)",
            implementation=AmplitudeLoadingImplementation.GRAYCODE,
        ),
        in_wires={"AMPLITUDE": qpe_out["PHASE_ESTIMATION"]},
    )

    # Step 4 Inverse QPE
    i_qpe_out = model_hhl.PhaseEstimation(
        params=qpe,
        is_inverse=True,
        release_by_inverse=True,
        in_wires={
            "PHASE_ESTIMATION": al_out["AMPLITUDE"],
            "OUT": qpe_out["OUT"],
        },
    )

    model_hhl.sample()

    model_hhl.set_outputs({"target": al_out["TARGET"], "solution": i_qpe_out["IN"]})

    # set Execution Preferences
    serialized_hhl_model = model_hhl.get_model()
    serialized_hhl_model = set_execution_preferences(
        serialized_hhl_model,
        execution_preferences=ExecutionPreferences(
            num_shots=1,
            backend_preferences=ClassiqBackendPreferences(
                backend_name="aer_simulator_statevector"
            ),
        ),
    )

    # Synth circuit
    qprog_hhl = synthesize(serialized_hhl_model)

    with open("hhl.qmod", "w") as f:
        f.write(serialized_hhl_model)

    return qprog_hhl, matrix_a, vector_b, w_min, qpe_register_size


def solve_hhl_classiq(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    qpe_register_size: Optional[int] = None,
    show_circuit: bool = False,
) -> Tuple[np.ndarray, str, int, int, float]:
    """Full implementation unified between classiq and qiskit."""
    np.set_printoptions(precision=3, suppress=True)
    start_time = time.time()

    circuit_hhl, _, _, w_min, qpe_register_size = classiq_hhl_implementation(
        matrix_a=matrix_a, vector_b=vector_b, qpe_register_size=qpe_register_size
    )
    if show_circuit:
        show(circuit_hhl)

    gen_circ = GeneratedCircuit.parse_raw(circuit_hhl)
    circuit_depth = gen_circ.transpiled_circuit.depth
    circuit_width = gen_circ.data.width
    print("depth = ", circuit_depth)
    print("width = ", circuit_width)

    qasm_content = gen_circ.transpiled_circuit.qasm

    # extract solution vector
    smallest_eigenval = min(np.linalg.eigvals(matrix_a))

    quantum_solution = extract_solution(
        qprog_hhl=circuit_hhl,
        w_min=w_min,
        matrix_a=matrix_a,
        vec_b=vector_b,
        smallest_ev=smallest_eigenval,
        qpe_register_size=qpe_register_size,
    )

    # todo: this actually might not the real runtime here, but includes the waiting time
    return (
        quantum_solution,
        qasm_content,
        circuit_depth,
        circuit_width,
        time.time() - start_time,
    )


if __name__ == "__main__":
    # input params
    N: int = 2

    model = ClassiqDemoExample()

    qsol, _, depth, width, run_time = solve_hhl_classiq(
        matrix_a=model.matrix_a, vector_b=model.vector_b, show_circuit=True
    )

    print_results(
        quantum_solution=qsol,
        classical_solution=model.classical_solution,
        run_time=run_time,
        name=model.name,
        plot=True,
    )
