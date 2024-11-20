"""VQLS implementation using Classiq"""
import numpy as np
from classiq import Model
from classiq.applications.chemistry import HVAParameters
from classiq.applications.combinatorial_optimization import (
    # QAOAConfig,
    OptimizerConfig,
)
from classiq.execution import OptimizerType

# from classiq import construct_combinatorial_optimization_model

matrix_a = np.random.rand(2, 2)


# hamiltonian variational ansatz
ansatz = HVAParameters(reps=3)
# state preparation of b
# matrix decomposition of A into linear combination of unitaries (vqls-prototype defaults to symmetric)

# optimizer to minimize cost function
optimizer_config = OptimizerConfig(opt_type=OptimizerType.COBYLA, max_iteration=250)
# Note : maybe see https://pennylane.ai/qml/demos/tutorial_vqls
hadamard_test_paulistring = 1
Model.vqe(
    hamiltonian=hadamard_test_paulistring,
    optimizer=optimizer_config,
    max_iteration=250,
    maximize=False,
    initial_point=ansatz,
)

# cost function
# global_cost_function
# local_cost_function
#
# # todo: fix this rest
#
# # Quantum circuit parameters
# num_qubits = A.shape[0]
# depth = 2
# num_iterations = 100
#
# # Prepare initial state
# initial_state = np.ones(num_qubits) / np.sqrt(num_qubits)
#
# # Create a quantum circuit (Classiq syntax might vary)
# qc = QuantumCircuit(num_qubits)
# qc.initialize(initial_state, range(num_qubits))
#
# # Variational form construction (Classiq syntax might vary)
# for d in range(depth):
#     for i in range(num_qubits):
#         qc.rx(qc.circuit_parameters[f'rx_{d}_{i}'], i)
#
# # Define an optimizer (Classiq syntax might vary)
# optimizer = Optimizer(max_iterations=num_iterations)
#
# # Define the Hamiltonian minimization problem (Classiq syntax might vary)
# hamiltonian_problem = HamiltonianMinimizationProblem(
#     ansatz=qc,  # Use your constructed quantum circuit
#     hamiltonian=A,  # Use your matrix A
#     target_vector=b,  # Use your vector b
#     optimizer=optimizer
# )
#
# # Execute the Hamiltonian minimization problem and obtain results (Classiq syntax might vary)
# result = hamiltonian_problem.execute()
#
# # Extract the solution vector
# solution_vector = result.optimal_parameters
# solution = np.dot(A, solution_vector)
#
# print("Solution vector:", solution_vector)
# print("Solution:", solution)
