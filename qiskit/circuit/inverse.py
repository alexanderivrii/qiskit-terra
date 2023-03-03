from qiskit.circuit import Operation, LazyOp


# ToDo: how do we make it really work on Operations? Need equality / inverses.
def are_inverse_ops(op1: Operation, op2: Operation) -> bool:

    # This can be improved in several ways
    if (op1.num_qubits != op2.num_qubits) or (op1.num_clbits != op2.num_clbits):
        return False

    # Case: both ops are not lazy
    if not isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):
        if getattr(op1, "inverse", None) is not None:
            return op1.inverse() == op2
        if getattr(op2, "inverse", None) is not None:
            return op2.inverse() == op1
        return False

    # Case: both ops are lazy
    if isinstance(op1, LazyOp) and isinstance(op2, LazyOp):
        if op1.num_ctrl_qubits != op2.num_ctrl_qubits:
            return False

        if (op1.inverted == op2.inverted) and are_inverse_ops(op1.base_op, op2.base_op):
            return True

        if (op1.inverted != op2.inverted) and (op1.base_op == op2.base_op):
            return True

        return False

    # Case: op1 is lazy, op2 is not
    if isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):
        if op1.num_ctrl_qubits != 0:
            return False
        if op1.inverted:
            return op1.base_op == op2
        else:
            return are_inverse_ops(op1.base_op, op2)

    if not isinstance(op1, LazyOp) and isinstance(op2, LazyOp):
        return are_inverse_ops(op2, op1)

    return False
