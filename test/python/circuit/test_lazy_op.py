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

from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Barrier, Measure, Reset, Gate, Operation, LazyOp
from qiskit.circuit.library import XGate, CXGate, SGate
from qiskit.quantum_info.operators import Clifford, CNOTDihedral, Pauli
from qiskit.extensions.quantum_initializer import Initialize, Isometry


class TestLazyOpClass(QiskitTestCase):
    """Testing qiskit.circuit.LazyOp"""

    def test_lazy_inverse(self):
        """Test that lazy inverse results in LazyOp."""
        gate = SGate()
        lazy_gate = gate.lazy_inverse()
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, SGate)

    def test_lazy_control(self):
        """Test that lazy control results in LazyOp."""
        gate = CXGate()
        lazy_gate = gate.lazy_control(2)
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, CXGate)

    def test_lazy_iterative(self):
        """Test that iteratively applying lazy inverse and control
        combines lazy modifiers."""
        lazy_gate = CXGate().lazy_inverse().lazy_control(2).lazy_inverse().lazy_control(1)
        self.assertIsInstance(lazy_gate, LazyOp)
        self.assertIsInstance(lazy_gate.base_op, CXGate)
        self.assertFalse(lazy_gate.inverted)
        self.assertEqual(lazy_gate.num_ctrl_qubits, 3)


if __name__ == "__main__":
    unittest.main()
