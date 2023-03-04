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

"""Recursively remove LazyOps from a circuit."""
from qiskit.circuit import LazyOp, Operation
from qiskit.circuit.add_control import add_control
from qiskit.transpiler.basepasses import TransformationPass


class UnrollLazy(TransformationPass):
    def __init__(self, target=None):
        """Unroll Lazy"""
        super().__init__()
        self.target = target

    def run(self, dag):
        new_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            unrolled_op = self._unroll_op(node.op)
            new_dag.apply_operation_back(unrolled_op, node.qargs, node.cargs)
        return new_dag

    def _unroll_op(self, op: Operation) -> Operation:

        if isinstance(op, LazyOp):
            unrolled_op = self._unroll_op(op.base_op)

            if op.num_ctrl_qubits > 0:
                unrolled_op = add_control(
                    operation=unrolled_op,
                    num_ctrl_qubits=op.num_ctrl_qubits,
                    label=None,
                    ctrl_state=None,
                )

            if op.inverted:
                # ToDo: what do we do for clifford or Operation without inverse method?
                unrolled_op = unrolled_op.real_inverse()

            return unrolled_op

        if getattr(op, "definition", None) is not None:
            new_definition = self._unroll_definition_circuit(op.definition)
            op.definition = new_definition

        return op

    def _unroll_definition_circuit(self, circuit):
        unrolled_circuit = circuit.copy_empty_like()

        for instruction in circuit:
            unrolled_op = self._unroll_op(instruction.operation)
            unrolled_circuit.append(instruction.replace(operation=unrolled_op))

        return unrolled_circuit
