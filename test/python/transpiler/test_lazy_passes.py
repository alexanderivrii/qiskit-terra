# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler passes that deal with linear functions."""


import unittest
import numpy as np
from qiskit import QuantumRegister, transpile
from qiskit.transpiler.passes.basis import UnrollLazy
from qiskit.transpiler.passes.optimization.optimize_lazy import OptimizeLazy
from test import combine

from ddt import ddt

from qiskit.circuit import QuantumCircuit, Qubit, Clbit, LazyOp
from qiskit.circuit.library import XGate
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.transpiler.passes.synthesis import (
    LinearFunctionsSynthesis,
    HighLevelSynthesis,
    LinearFunctionsToPermutations,
)
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.circuit.library import RealAmplitudes, CCXGate
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator
from qiskit.circuit.library.basis_change import QFT


class TestLazyPasses(QiskitTestCase):

    @staticmethod
    def controlled_qft_adder(num_qubits, num_controls):
        """
        Creates a controlled QFT adder (code adapted from
        draper_qft_adder.py).
        """

        qr_a = QuantumRegister(num_qubits, name="a")
        qr_b = QuantumRegister(num_qubits, name="b")
        qr_z = QuantumRegister(1, name="cout")
        qr_list = [qr_a, qr_b, qr_z]

        circuit = QuantumCircuit(*qr_list, name="QFT_ADDER")

        qr_sum = qr_b[:] + qr_z[:]
        num_qubits_qft = num_qubits + 1

        circuit.append(QFT(num_qubits_qft, do_swaps=False).to_gate(), qr_sum[:])

        for j in range(num_qubits):
            for k in range(num_qubits - j):
                lam = np.pi / (2**k)
                circuit.cp(lam, qr_a[j], qr_b[j + k])

        for j in range(num_qubits):
            lam = np.pi / (2 ** (j + 1))
            circuit.cp(lam, qr_a[num_qubits - j - 1], qr_z[0])

        circuit.append(QFT(num_qubits_qft, do_swaps=False).to_gate().inverse(), qr_sum[:])

        if num_controls > 0:
            circuit = circuit.control(num_controls)

        return circuit

    @staticmethod
    def optimized_controlled_qft_adder(num_qubits, num_controls):
        """
        Optimized controlled QFT adder.
        Code adapted from draper_qft_adder.py.
        """

        qr_a = QuantumRegister(num_qubits, name="a")
        qr_b = QuantumRegister(num_qubits, name="b")
        qr_z = QuantumRegister(1, name="cout")

        if num_controls == 0:
            qr_list = [qr_a, qr_b, qr_z]

        else:
            qr_c = QuantumRegister(num_controls, name="cntl")
            qr_list = [qr_c, qr_a, qr_b, qr_z]

        circuit = QuantumCircuit(*qr_list, name="QFT_ADDER")

        qr_sum = qr_b[:] + qr_z[:]
        num_qubits_qft = num_qubits + 1

        circuit.append(QFT(num_qubits_qft, do_swaps=False).to_gate(), qr_sum[:])

        qri_list = [qr_a, qr_b, qr_z]

        inner_circuit = QuantumCircuit(*qri_list, name="INNER_CIRCUIT")
        for j in range(num_qubits):
            for k in range(num_qubits - j):
                lam = np.pi / (2**k)
                inner_circuit.cp(lam, qr_a[j], qr_b[j + k])

        for j in range(num_qubits):
            lam = np.pi / (2 ** (j + 1))
            inner_circuit.cp(lam, qr_a[num_qubits - j - 1], qr_z[0])

        if num_controls > 0:
            inner_circuit = inner_circuit.control(num_controls)

        circuit.append(inner_circuit, range(circuit.num_qubits))

        circuit.append(QFT(num_qubits_qft, do_swaps=False).inverse().to_gate(), qr_sum[:])

        return circuit

    @staticmethod
    def lazy_controlled_qft_adder(num_qubits, num_controls):
        """
        Code adapted from draper_qft_adder.py.
        """

        qr_a = QuantumRegister(num_qubits, name="a")
        qr_b = QuantumRegister(num_qubits, name="b")
        qr_z = QuantumRegister(1, name="cout")
        qr_list = [qr_a, qr_b, qr_z]

        circuit = QuantumCircuit(*qr_list, name="QFT_ADDER")

        qr_sum = qr_b[:] + qr_z[:]
        num_qubits_qft = num_qubits + 1
        qft = QFT(num_qubits_qft, do_swaps=False).to_gate()
        circuit.append(qft, qr_sum[:])

        for j in range(num_qubits):
            for k in range(num_qubits - j):
                lam = np.pi / (2**k)
                circuit.cp(lam, qr_a[j], qr_b[j + k])

        for j in range(num_qubits):
            lam = np.pi / (2 ** (j + 1))
            circuit.cp(lam, qr_a[num_qubits - j - 1], qr_z[0])

        qfti = QFT(num_qubits_qft, do_swaps=False).to_gate().lazy_inverse()
        circuit.append(qfti, qr_sum[:])

        if num_controls > 0:
            circuit = circuit.lazy_control(num_controls)

        return circuit

    @staticmethod
    def transpile_circuit(circuit, basis_gates, optimization_level):
        """Standard transpile (does not work on lazy circuits)"""
        transpiled_circuit = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=1,
        )
        return transpiled_circuit

    @staticmethod
    def transpile_lazy_circuit(circuit, basis_gates, optimization_level):
        """Unrolls and transpiles lazy circuits"""
        unrolled_circuit = UnrollLazy()(circuit)
        transpiled_circuit = transpile(
            unrolled_circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=1,
        )
        return transpiled_circuit

    @staticmethod
    def transpile_optimize_lazy_circuit(circuit, basis_gates, optimization_level):
        """Optimizes, unrolls and transpiles lazy circuits."""
        opt_circuit = OptimizeLazy()(circuit)
        unrolled_circuit = UnrollLazy()(opt_circuit)
        transpiled_circuit = transpile(
            unrolled_circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=1,
        )
        return transpiled_circuit

    def test_adder_equivalence(self):
        """Test that all adders are equivalent (after transpilation).
        Also see the effect of OptimizeLazy() reduction.
        """
        # basis_gates = ['cx', 'id', 'u']
        basis_gates = ['rz', 'sx', 'x', 'cx']
        num_qubits = 3
        num_controls = 2
        optimization_level = 1

        t1 = self.transpile_circuit(
            self.controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        t2 = self.transpile_optimize_lazy_circuit(
            self.controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        t3 = self.transpile_circuit(
            self.optimized_controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        t4 = self.transpile_optimize_lazy_circuit(
            self.optimized_controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        t5 = self.transpile_lazy_circuit(
            self.lazy_controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        t6 = self.transpile_optimize_lazy_circuit(
            self.lazy_controlled_qft_adder(num_qubits, num_controls),
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )

        print(t1.count_ops())
        print(t2.count_ops())
        print(t3.count_ops())
        print(t4.count_ops())
        print(t5.count_ops())
        print(t6.count_ops())

        self.assertEqual(Operator(t1), Operator(t2))
        self.assertEqual(Operator(t1), Operator(t3))
        self.assertEqual(Operator(t1), Operator(t4))
        self.assertEqual(Operator(t1), Operator(t5))
        self.assertEqual(Operator(t1), Operator(t6))

    def test_inverse_cancellation(self):
        """Test inverse cancellation with lazy gates."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.append(CCXGate().lazy_inverse(), [0, 1, 2])
        print(qc)
        qct = OptimizeLazy()(qc)
        print(qct)
        self.assertEqual(qct.size(), 0)

    def test_unroll_with_open_control(self):
        base_gate = XGate()
        num_ctrl_qubits = 3
        num_qubits = base_gate.num_qubits + num_ctrl_qubits

        for ctrl_state in [5, None, 0, 7, "110"]:
            qc = QuantumCircuit(num_qubits)
            lazy_gate = LazyOp(base_op=base_gate, num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
            qc.append(lazy_gate, range(num_qubits))
            qct = UnrollLazy()(qc)
            self.assertEqual(Operator(qc), Operator(qct))


if __name__ == "__main__":
    unittest.main()
