from typing import Optional, Union

from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int


class LazyOp(Operation):
    """Gate and modifiers inside."""

    def __init__(
        self,
        base_op,
        num_ctrl_qubits=0,
        ctrl_state: Optional[Union[int, str]] = None,
        inverted=False,
        label: Optional[str] = None,
    ):
        self.base_op = base_op
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        self.inverted = inverted
        self._name = "lazy"
        self.label = label

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
            ctrl_state=self.ctrl_state,
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

        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_num_ctrl_qubits = self.num_ctrl_qubits + num_ctrl_qubits
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        return LazyOp(
            self.base_op,
            num_ctrl_qubits=new_num_ctrl_qubits,
            ctrl_state=new_ctrl_state,
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

        from qiskit.circuit.inverse import are_equal_ops
        return are_equal_ops(self, other)

    def print_rec(self, offset=0, depth=100, header=""):
        """Temporary debug function."""
        line = " " * offset + header + \
               " LazyGate " + self.name + \
               "[ c" + str(self.num_ctrl_qubits) + " inv" + str(self.inverted) + "]"
        print(line)
        if depth >= 0:
            self.base_op.print_rec(offset + 2, depth - 1, header="base gate")

    def copy(self) -> "LazyOp":
        """Return a copy of the :class:`LazyOp`."""
        return LazyOp(
            base_op=self.base_op.copy(),
            num_ctrl_qubits=self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state,
            inverted=self.inverted,
        )

    def to_matrix(self):
        """Return a matrix representation (allowing to construct Operator)."""
        from qiskit.quantum_info import Operator

        operator = Operator(self.base_op)

        if self.inverted:
            operator = operator.power(-1)

        return _compute_control_matrix(operator.data, self.num_ctrl_qubits, self.ctrl_state)

    @property
    def definition(self):
        """
        Question: do we want lazy ops to have the definition function?
        """
        from qiskit.transpiler.passes.basis.unroll_lazy import UnrollLazy

        unrolled_op = UnrollLazy()._unroll_op(self)
        return unrolled_op.definition

