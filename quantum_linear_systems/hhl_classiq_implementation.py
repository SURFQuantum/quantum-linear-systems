import time
import numpy as np
from itertools import product

from classiq.builtin_functions import StatePreparation, Exponentiation, PhaseEstimation, AmplitudeLoading
from classiq.builtin_functions.exponentiation import PauliOperator
from classiq.interface.generator.qpe import (
    ExponentiationScaling,
    ExponentiationSpecification,
)
from classiq.interface.generator.amplitude_loading import AmplitudeLoadingImplementation
from classiq import Model, execute, synthesize, show, GeneratedCircuit
from classiq.execution import ExecutionPreferences, IBMBackendPreferences
from classiq.synthesis import set_execution_preferences
from classiq.execution import ExecutionDetails

from quantum_linear_systems.toymodels import classiq_demo_problem
from quantum_linear_systems.utils import extract_x_from_expanded, plot_csol_vs_qsol


Paulidict = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
}


# generate all combinations of Pauli strings of size n
def generate_all_pauli_strings(seq, n):
    """
    Generate all combinations of Pauli strings of size n.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).
        n (int): The size of the output Pauli strings.

    Returns:
        A generator of all combinations of Pauli strings of size n.
    """
    for s in product(seq, repeat=n):
        yield "".join(s)


# convert a Paulistring of size n to 2**n X 2**n matrix
def pauli_string_2mat(seq):
    """
    Convert a Pauli string of size n to a 2**n x 2**n matrix.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).

    Returns:
        A 2**n x 2**n matrix representation of the Pauli string.
    """
    p_matrix = Paulidict[seq[0]]
    for p in seq[1:]:
        p_matrix = np.kron(p_matrix, Paulidict[p])
    return p_matrix


# Hilbert-Schmidt-Product of two matrices M1, M2
def hilbert_schmidt(m1, m2):
    """
    Compute the Hilbert-Schmidt-Product of two matrices M1, M2.

    Parameters:
        m1 (np.ndarray): The first matrix.
        m2 (np.ndarray): The second matrix.

    Returns:
        The Hilbert-Schmidt inner product of the two matrices.
    """
    return (np.dot(m1.conjugate().transpose(), m2)).trace()


# Naive decomposition, running over all HS products for all Pauli strings
def lcu_naive(hm):
    """
    Naive LCU (linear combination of unitary operations) decomposition, running over all HS products for all Pauli
    strings.

    Parameters:
        hm (np.ndarray): The input Hermitian matrix.

    Returns:
        A list of tuples, each containing a Pauli string and the corresponding coefficient.
    """
    assert hm.shape[0] == hm.shape[1], "matrix is not square"
    assert hm.shape[0] != 0, "matrix is of size 0"
    assert hm.shape[0] & (hm.shape[0] - 1) == 0, "matrix size is not 2**n"

    n = int(np.log2(hm.shape[0]))
    pauli_strings = list(generate_all_pauli_strings("IZXY", n))

    mylist = []

    for pstr in pauli_strings:
        co = (1 / 2**n) * hilbert_schmidt(pauli_string_2mat(pstr), hm)
        if co != 0:
            mylist = mylist + [(pstr, co)]

    return mylist


def state_preparation(vector_b, sp_upper):
    """
    Prepare the state based on the input vector.

    Parameters:
        vector_b (np.ndarray): The input vector.
        sp_upper (float): The upper bound of the L2 error metric for the state preparation.

    Returns:
        A StatePreparation object with the desired amplitudes and error metric.
    """
    vector_b = tuple(vector_b)
    # sp_upper = precision of the State Preparation
    return StatePreparation(
        amplitudes=vector_b, error_metric={"L2": {"upper_bound": sp_upper}})


def quantum_phase_estimation(paulis, precision):
    """
    Perform Quantum Phase Estimation (QPE) with the specified precision.

    Parameters:
        precision (int): The desired precision for the QPE.

    Returns:
        A PhaseEstimation object configured with the specified precision.
    """
    po = PauliOperator(pauli_list=paulis)
    exp_params = Exponentiation(
        pauli_operator=po,
        evolution_coefficient=-2 * np.pi,
    )

    return PhaseEstimation(
        size=precision,
        unitary_params=exp_params,
        exponentiation_specification=ExponentiationSpecification(
            scaling=ExponentiationScaling(max_depth=100, max_depth_scaling_factor=2)
        ),
    )


def verification_of_result(circuit, w_min, sol_classical):
    """
    Verify the result of the quantum algorithm by comparing with the classical solution.

    Parameters:
        circuit (str): The quantum circuit.
        w_min
        sol_classical (np.ndarray) : classical solution vector

    Returns:
        A tuple containing the classical solution and the solution obtained from the quantum algorithm.
    """
    results = execute(circuit)
    res_hhl = ExecutionDetails.parse_obj(results[0].value)
    circuit = GeneratedCircuit.parse_raw(circuit)

    # qsol_pure= Statevector(circuit).data
    # qsol_pure = [0] * 4

    total_q = circuit.data.width  # total number of qubits of the whole circuit

    target_pos = res_hhl.output_qubits_map["target"][0]  # position of control qubit

    sol_pos = list(res_hhl.output_qubits_map["solution"])  # position of solution

    canonical_list = np.array(list("0" * total_q))  # we start with a string of zeros
    canonical_list[
        target_pos
    ] = "1"  # we are interested in strings having 1 on their target qubit

    qsol = list()
    for i in range(2 ** len(sol_pos)):
        templist = canonical_list.copy()
        templist[sol_pos] = list(np.binary_repr(i, len(sol_pos))[::-1])
        qsol.append(np.round(complex(res_hhl.state_vector["".join(templist)]) / w_min, 5))

    if len(qsol) > len(sol_classical):
        qsol = extract_x_from_expanded(np.array(qsol))  # extract the solution from the extended vector

    print("first", qsol)
    global_phase = np.angle(qsol)
    qsol_corrected = np.real(qsol / np.exp(1j * global_phase))
    print("classical:  ", sol_classical)
    print("HHL:        ", qsol_corrected)
    print(
        "relative distance:  ",
        round(
            np.linalg.norm(sol_classical - qsol_corrected) / np.linalg.norm(sol_classical) * 100, 1,
        ),
        "%",
    )
    return qsol_corrected
    # return qsol_corrected, qsol_pure


def verify_matrix_sym_and_pos_ev(mat):
    """
    Verify that the input matrix is symmetric and has positive eigenvalues.

    Parameters:
        mat (np.ndarray): The input matrix.
    """
    if not np.allclose(mat, mat.T, rtol=1e-6, atol=1e-6):
        raise Exception("The matrix is not symmetric")
    w, v = np.linalg.eig(mat)
    for lam in w:
        if lam < 0:
            raise Exception("The matrix has negative eigenvalues")
        # the original classiq workshop checked if the EV are in [0,1)
        if lam > 1:
            print("The matrix has eigenvalues larger than 1: ", lam)


def classiq_hhl_implementation(matrix_a, vector_b, precision):
    # verifying that the matrix is symmetric and hs eigenvalues in [0,1)
    # verify_matrix_sym_and_pos_ev(mat=matrix_a)

    paulis = lcu_naive(matrix_a)
    # print("Pauli strings list: \n")
    # for p in paulis:
    #     print(p[0], ": ", np.round(p[1], 3))

    print("\n Number of qubits for matrix representation =", len(paulis[0][0]))

    # Step 1: state preparation
    if np.linalg.norm(vector_b) != 1:
        print(f"Normalizing A and b by {np.linalg.norm(vector_b)}")
    b_normalized = vector_b / np.linalg.norm(vector_b)
    A_normalized = matrix_a / np.linalg.norm(vector_b)

    sp = state_preparation(vector_b=b_normalized, sp_upper=0.00)

    # Step 2 : Quantum Phase Estimation
    qpe = quantum_phase_estimation(paulis=paulis, precision=precision)

    # Step 3 : Eigenvalue Inversion

    w_min = (1 / 2 ** precision)  # for qpe register of size m, this is the minimal value which can be encoded
    expression = f"{w_min}/(x)"
    al_params = AmplitudeLoading(
        size=precision,
        expression=expression,
        implementation=AmplitudeLoadingImplementation.GRAYCODE,
    )

    # Step 4 Inverse QPE

    model_hhl = Model()
    sp_out = model_hhl.StatePreparation(params=sp)
    qpe_out = model_hhl.PhaseEstimation(params=qpe, in_wires={"IN": sp_out["OUT"]})
    al_out = model_hhl.AmplitudeLoading(
        params=al_params,
        in_wires={"AMPLITUDE": qpe_out["PHASE_ESTIMATION"]},
    )
    i_qpe_out = model_hhl.PhaseEstimation(
        params=qpe,
        is_inverse=True,
        release_by_inverse=True,
        in_wires={
            "PHASE_ESTIMATION": al_out["AMPLITUDE"],
            "OUT": qpe_out["OUT"],
        },
    )

    model_hhl.set_outputs({"target": al_out["TARGET"], "solution": i_qpe_out["IN"]})

    # set Execution Preferences
    backend_preferences = IBMBackendPreferences(
        backend_service_provider="IBM Quantum", backend_name="aer_simulator_statevector"
    )

    serialized_hhl_model = model_hhl.get_model()

    serialized_hhl_model = set_execution_preferences(
        serialized_hhl_model,
        execution_preferences=ExecutionPreferences(
            num_shots=1, backend_preferences=backend_preferences
        ),
    )

    # Synth circuit
    qprog_hhl = synthesize(serialized_hhl_model)

    return qprog_hhl, A_normalized, b_normalized, w_min


if __name__ == "__main__":
    start_time = time.time()

    # input params
    n = 2
    precision = 4

    # A, b, csol, name = volterra_problem(n)
    A, b, csol, name = classiq_demo_problem()

    circuit_hhl, A_normalized, b_normalized, w_min = classiq_hhl_implementation(matrix_a=A, vector_b=b,
                                                                                precision=precision)
    show(circuit_hhl)

    gen_circ = GeneratedCircuit.parse_raw(circuit_hhl)
    print("depth = ", gen_circ.transpiled_circuit.depth)

    # verify against classical solution
    qsol = verification_of_result(circuit=circuit_hhl, w_min=w_min, sol_classical=csol)

    csol /= np.linalg.norm(csol)
    qsol /= np.linalg.norm(qsol)
    print("classical", csol.flatten())
    print("quantum", qsol.flatten())

    plot_csol_vs_qsol(csol, qsol, "Classiq")

    print(f"Finished classiq run in {time.time() - start_time}s.")

    if np.linalg.norm(csol - qsol) / np.linalg.norm(csol) > 0.2:
        raise RuntimeError("The HHL solution is too far from the classical one, please verify your algorithm.")
