from qiskit.circuit import Operation, LazyOp

# ToDo: this is absolutely horrible!
# But in order to support gate.control().inverse() == gate.inverse().control(),
# we need to consider cases where both gates are lazy, both gates are not lazy,
# one gate is lazy and one is not.


def are_inverse_ops(op1: Operation, op2: Operation) -> bool:

    if (op1.num_qubits != op2.num_qubits) or (op1.num_clbits != op2.num_clbits):
        return False

    # Case 1: both ops are not lazy
    if not isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):

        if getattr(op1, "inverse", None) is not None:
            op1_inverse = op1.inverse()
            if isinstance(op1_inverse, LazyOp):
                op1_inverse = op1.real_inverse()
            assert not isinstance(op1_inverse, LazyOp)
            return op1_inverse == op2

        if getattr(op2, "inverse", None) is not None:
            op2_inverse = op2.inverse()
            if isinstance(op2_inverse, LazyOp):
                op2_inverse = op2.real_inverse()
            assert not isinstance(op2_inverse, LazyOp)
            return op2_inverse == op1

        return False

    # Case 2: both ops are lazy
    if isinstance(op1, LazyOp) and isinstance(op2, LazyOp):

        if op1.num_ctrl_qubits != op2.num_ctrl_qubits:
            return False
        if op1.ctrl_state != op2.ctrl_state:
            return False

        if (op1.inverted == op2.inverted) and are_inverse_ops(op1.base_op, op2.base_op):
            return True

        if (op1.inverted != op2.inverted) and are_equal_ops(op1.base_op, op2.base_op):
            return True

        return False

    # Case 3: op1 is lazy, op2 is not
    if isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):
        if op1.num_ctrl_qubits != 0:
            return False
        if not op1.inverted:
            return are_inverse_ops(op1.base_op, op2)
        else:
            return are_equal_ops(op1.base_op, op2)

    # Case 4: op2 is lazy, op1 is not
    if isinstance(op2, LazyOp) and not isinstance(op1, LazyOp):
        if op2.num_ctrl_qubits != 0:
            return False
        if not op2.inverted:
            return are_inverse_ops(op2.base_op, op1)
        else:
            return are_equal_ops(op2.base_op, op1)

    # Actually, we should not be here
    return False


def are_equal_ops(op1: Operation, op2: Operation) -> bool:

    if (op1.num_qubits != op2.num_qubits) or (op1.num_clbits != op2.num_clbits):
        return False

    # Case 1: both ops are not lazy, we default to the standard equality between such gates
    if not isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):
        return op1 == op2

    # Case 2: both ops are lazy
    if isinstance(op1, LazyOp) and isinstance(op2, LazyOp):

        if op1.num_ctrl_qubits != op2.num_ctrl_qubits:
            return False
        if op1.ctrl_state != op2.ctrl_state:
            return False

        if (op1.inverted == op2.inverted) and are_equal_ops(op1.base_op, op2.base_op):
            return True

        if (op1.inverted != op2.inverted) and are_inverse_ops(op1.base_op, op2.base_op):
            return True

        return False

    # Case 3: op1 is lazy, op2 is not
    if isinstance(op1, LazyOp) and not isinstance(op2, LazyOp):
        if op1.num_ctrl_qubits != 0:
            return False
        if not op1.inverted:
            return are_equal_ops(op1.base_op, op2)
        else:
            return are_inverse_ops(op1.base_op, op2)

    # Case 4: op2 is lazy, op1 is not
    if isinstance(op2, LazyOp) and not isinstance(op1, LazyOp):
        if op2.num_ctrl_qubits != 0:
            return False
        if not op2.inverted:
            return are_equal_ops(op2.base_op, op1)
        else:
            return are_inverse_ops(op2.base_op, op1)

    # Actually, we should not be here
    return False
