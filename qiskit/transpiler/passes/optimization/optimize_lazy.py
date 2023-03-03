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

"""Recursively optimizes circuits with lazy ops."""

from typing import Union
from qiskit.circuit import QuantumCircuit, Gate, Operation, LazyOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.inverse import are_inverse_ops
from qiskit.converters import dag_to_circuit, circuit_to_dag


class OptimizeLazy(TransformationPass):
    """Optimization pass on circuits with lazy ops."""

    def __init__(self, target=None):
        """
        Optimization pass on circuits with lazy ops.
        We probably need additional arguments to specify which optimizations to perform.
        """

        super().__init__()
        self._target = target

    def run(self, dag):
        """Run the OptimizeLazy pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Returns:
            DAGCircuit: output optimized dag
        """

        dag = self._inverse_cancellation(dag)
        dag = self._conjugate_reduction(dag)

        # Recursively optimize definitions
        for node in dag.op_nodes():
            self._optimize_op_definition_circuit(node.op)
        return dag

    @staticmethod
    def _inverse_cancellation(dag):
        """Simple pass to go over DAG, removing pairs of consecutive inverse ops"""

        def _skip_node(node):
            """Returns True if we should skip this node for the analysis."""

            if getattr(node.op, "_directive", False) or node.name in {"measure", "reset", "delay"}:
                return True
            if getattr(node.op, "condition", None):
                return True
            return False

        topo_sorted_nodes = list(dag.topological_op_nodes())
        circ_size = len(topo_sorted_nodes)
        removed = [False for _ in range(circ_size)]

        # Go over DAG, see which nodes can be removed
        for idx in range(0, circ_size - 1):
            if removed[idx] or removed[idx + 1]:
                continue

            node1 = topo_sorted_nodes[idx]
            node2 = topo_sorted_nodes[idx + 1]

            if _skip_node(node1) or _skip_node(node2):
                continue

            if node1.qargs != node2.qargs or node1.cargs != node2.cargs:
                continue

            if are_inverse_ops(node1.op, node2.op):
                print("=> INVERSE REDUCTION")
                removed[idx] = True
                removed[idx + 1] = True

        # Actually remove nodes
        for idx in range(circ_size):
            if removed[idx]:
                dag.remove_op_node(topo_sorted_nodes[idx])

        return dag

    def _conjugate_reduction(self, dag):
        for node in dag.op_nodes():
            if isinstance(node.op, LazyOp):
                optimized_op = self._lazy_op_conjugate_reduction(node.op)
                node.op = optimized_op
        return dag

    @staticmethod
    def _split_by_conjugation(circuit):
        """
        Given a quantum circuit, check if it's of the form PQP^{-1}.
        If so, returns triple (P, Q, P^{-1}).
        If not, returns triple (None, circuit, None).
        """
        num_matched = 0

        for idx in range(len(circuit.data) // 2):
            fwd_instruction = circuit.data[idx]
            bwd_instruction = circuit.data[-idx - 1]
            if (
                (fwd_instruction.qubits == bwd_instruction.qubits)
                and (fwd_instruction.clbits == bwd_instruction.clbits)
                and (are_inverse_ops(fwd_instruction.operation, bwd_instruction.operation))
            ):
                num_matched += 1
            else:
                break

        if num_matched == 0:
            return None, circuit, None

        else:
            prefix_circuit = circuit.copy_empty_like()
            middle_circuit = circuit.copy_empty_like()
            suffix_circuit = circuit.copy_empty_like()

            prefix_circuit.data = circuit.data[0:num_matched]
            middle_circuit.data = circuit.data[num_matched:-num_matched]
            suffix_circuit.data = circuit.data[-num_matched:]

            # ToDo: fix non-global-phases!!!
            return prefix_circuit, middle_circuit, suffix_circuit

    def _lazy_op_conjugate_reduction(self, op: LazyOp) -> Union[LazyOp, Gate]:
        """
        Optimizes a lazy op.

        Suppose we have a LazyOp L, with the base gate B having definition P Q P^{-1}.

        We have:
        control-[PQP^{-1}] = [control-P][control-Q][control-P^{-1}] = P[control-Q]P^{-1}.
        [PQP^{-1}]^{-1} = P[Q^{-1}]P^{-1}.

        (If control=1, then both sides are PQP^{-1}. If control=0, then both sides are Id.)

        That is, both control and inverse modifiers descend onto Q.

        The reduction produces a new gate (using circuit.to_gate()) with def = P M P^{-1}, where
        M is a lazy gate with original control and inverse modifiers, and base gate Q.to_gate().
        """
        base_op = op.base_op
        if getattr(base_op, "definition", None) is None:
            return op
        definition_circuit = base_op.definition
        prefix_circuit, middle_circuit, suffix_circuit = self._split_by_conjugation(definition_circuit)
        if prefix_circuit is None:
            return op

        new_definition_circuit = QuantumCircuit(op.num_qubits, op.num_clbits)

        def _map_qubits(qubits):
            return [op.num_ctrl_qubits + definition_circuit.find_bit(q).index for q in qubits]

        for circuit_instruction in prefix_circuit.data:
            mapped_qubits = _map_qubits(circuit_instruction.qubits)

            new_definition_circuit.append(
                circuit_instruction.operation,
                mapped_qubits,
                circuit_instruction.clbits,
            )

        m_base_op = middle_circuit.to_gate()
        m_op = LazyOp(
            base_op=m_base_op, num_ctrl_qubits=op.num_ctrl_qubits, inverted=op.inverted
        )
        new_definition_circuit.append(m_op, range(op.num_qubits), cargs=range(op.num_clbits))
        for circuit_instruction in suffix_circuit.data:
            new_definition_circuit.append(
                circuit_instruction.operation,
                _map_qubits(circuit_instruction.qubits),
                circuit_instruction.clbits,
            )
        new_op = new_definition_circuit.to_gate()
        return new_op

    def _optimize_op_definition_circuit(self, op: Operation):
        """Recursively optimizes definition circuit"""

        if isinstance(op, LazyOp):
            if getattr(op.base_op, "definition", None) is not None:
                op.base_op.definition = self._optimize_definition_circuit(
                    op.base_op.definition
                )
        else:
            if getattr(op, "definition", None) is not None:
                op.definition = self._optimize_definition_circuit(op.definition)
        return op

    def _optimize_definition_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Recursive circuit optimization."""
        dag = circuit_to_dag(circuit)
        dag = self.run(dag)
        optimized_circuit = dag_to_circuit(dag)
        return optimized_circuit
