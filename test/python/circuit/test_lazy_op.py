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

"""Test Qiskit's LazyOp class."""

import unittest

import numpy as np

from qiskit.circuit._utils import _compute_control_matrix
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Barrier, Measure, Reset, Gate, Operation, LazyOp
from qiskit.circuit.library import XGate, CXGate, SGate, SwapGate, U2Gate
from qiskit.quantum_info.operators import Clifford, CNOTDihedral, Pauli
from qiskit.extensions.quantum_initializer import Initialize, Isometry
from qiskit.quantum_info import Operator


class TestLazyOpClass(QiskitTestCase):
    """Testing qiskit.circuit.LazyOp"""

    def _test_lazy_inverse(self):
        """Test that lazy inverse results in LazyOp."""
        gate = SGate()
        lazy_gate = gate.lazy_inverse()
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, SGate)

    def _test_lazy_control(self):
        """Test that lazy control results in LazyOp."""
        gate = CXGate()
        lazy_gate = gate.lazy_control(2)
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, CXGate)

    def _test_lazy_iterative(self):
        """Test that iteratively applying lazy inverse and control
        combines lazy modifiers."""
        lazy_gate = CXGate().lazy_inverse().lazy_control(2).lazy_inverse().lazy_control(1)
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, CXGate)
        self.assertFalse(lazy_gate.inverted)
        self.assertEqual(lazy_gate.num_ctrl_qubits, 3)

    def _test_lazy_open_control(self):
        base_gate = XGate()
        base_mat = base_gate.to_matrix()
        num_ctrl_qubits = 3

        for ctrl_state in [5, None, 0, 7, "110"]:
            lazy_gate = LazyOp(base_op=base_gate, num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
            target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state)
            self.assertEqual(Operator(lazy_gate), Operator(target_mat))

    def _test_definition(self):
        base_gate = SwapGate()
        lazy_gate = LazyOp(base_op=base_gate, num_ctrl_qubits=2, inverted=False)
        lazy_gate_definition = lazy_gate.definition
        self.assertEqual(Operator(lazy_gate), Operator(lazy_gate_definition))


    def test_equality(self):
        base_gate = U2Gate(phi=2, lam=2)
        print(base_gate)
        gate_i = base_gate.inverse()
        print(f"{gate_i = }")
        gate_ii = gate_i.inverse()
        print(f"{gate_ii = }")

        gate_c = base_gate.control()
        print(f"{gate_c = }")
        gate_ic = gate_i.control()
        print(f"{gate_ic = }")

        gate_ci = gate_c.inverse()
        print(f"{gate_ci = }")

        print(f"{gate_ic.base_op = }, {gate_ic.inverted = }, {gate_ic.num_ctrl_qubits = }, {gate_ic.ctrl_state = }")
        print(f"{gate_ci.base_op = }, {gate_ci.inverted = }, {gate_ci.num_ctrl_qubits = }, {gate_ci.ctrl_state = }")

        # eq = gate_ic == gate_ci
        from qiskit.circuit.inverse import are_inverse_ops, are_equal_ops
        eq = are_equal_ops(gate_ic, gate_ci)
        print(f"{eq = }")



if __name__ == "__main__":
    unittest.main()
