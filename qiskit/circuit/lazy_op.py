from typing import Optional, Union

from qiskit.circuit.operation import Operation


class LazyOp(Operation):
    """Gate and modifiers inside."""

    def __init__(
        self,
        base_op,
        num_ctrl_qubits=0,
        inverted=False,
    ):
        self.base_op = base_op
        self.num_ctrl_qubits = num_ctrl_qubits
        self.inverted = inverted
        self._name = "lazy"

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def num_qubits(self):
        """Number of qubits."""
        return self.num_ctrl_qubits + self.base_op.num_qubits

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return self.base_op.num_clbits

    def lazy_inverse(self):
        """Returns lazy inverse
        Maybe does not belong here
        """

        # ToDo: Should we copy base_op?
        return LazyOp(
            self.base_op,
            num_ctrl_qubits=self.num_ctrl_qubits,
            inverted=not self.inverted,
        )

    def inverse(self):
        return self.lazy_inverse()

    def lazy_control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Maybe does not belong here"""

        return LazyOp(
            self.base_op,
            num_ctrl_qubits=self.num_ctrl_qubits + num_ctrl_qubits,
            inverted=self.inverted,
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        return self.lazy_control(num_ctrl_qubits, label, ctrl_state)

    def __eq__(self, other) -> bool:
        """Checks if two LazyOps are equal."""
        return (
            isinstance(other, LazyOp)
            and self.num_ctrl_qubits == other.num_ctrl_qubits
            and self.num_ctrl_qubits == other.num_ctrl_qubits
            and self.inverted == other.inverted
            and self.base_op == other.base_op
        )

    def print_rec(self, offset=0, depth=100, header=""):
        """Temporary debug function."""
        line = (
            " " * offset + header + " LazyGate " + self.name + "["
            " c" + str(self.num_ctrl_qubits) + " p" + str(self.inverted) + "]"
        )
        print(line)
        if depth >= 0:
            self.base_op.print_rec(offset + 2, depth - 1, header="base gate")

    def copy(self) -> "LazyOp":
        """Return a copy of the :class:`LazyOp`."""
        return LazyOp(
            base_op=self.base_op.copy(),
            num_ctrl_qubits=self.num_ctrl_qubits,
            inverted=self.inverted,
        )

    def to_matrix(self):
        """Return a matrix representation (allowing to construct Operator)."""
        import numpy as np
        from qiskit.quantum_info import Operator

        operator = Operator(self.base_op)

        if self.inverted:
            operator = operator.power(-1)

        for _ in range(self.num_ctrl_qubits):
            dim = int(np.log2(operator._input_dim))
            op0 = Operator(np.eye(2 ** dim)).tensor([[1, 0], [0, 0]])
            op1 = operator.tensor([[0, 0], [0, 1]])
            operator = op0 + op1

        return operator.data
