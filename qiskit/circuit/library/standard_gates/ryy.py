# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit YY-rotation gate."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit._accelerate.circuit import StandardGate


class RYYGate(Gate):
    r"""A parametric 2-qubit :math:`Y \otimes Y` interaction (rotation about YY).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ryy` method.

    **Circuit Symbol:**

    .. code-block:: text

             ┌─────────┐
        q_0: ┤1        ├
             │  Ryy(ϴ) │
        q_1: ┤0        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{YY}(\theta) = \exp\left(-i \rotationangle Y{\otimes}Y\right) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & 0 & 0 & i\sin\left(\rotationangle\right) \\
                0 & \cos\left(\rotationangle\right) & -i\sin\left(\rotationangle\right) & 0 \\
                0 & -i\sin\left(\rotationangle\right) & \cos\left(\rotationangle\right) & 0 \\
                i\sin\left(\rotationangle\right) & 0 & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{YY}(\theta = 0) = I

        .. math::

            R_{YY}(\theta = \pi) = -i Y \otimes Y

        .. math::

            R_{YY}\left(\theta = \frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1 & 0 & 0 & i \\
                                        0 & 1 & -i & 0 \\
                                        0 & -i & 1 & 0 \\
                                        i & 0 & 0 & 1
                                    \end{pmatrix}
    """

    _standard_gate = StandardGate.RYY

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new RYY gate."""
        super().__init__("ryy", 2, [theta], label=label)

    def _define(self):
        """Default definition"""
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        #      ┌──────┐                   ┌────┐
        # q_0: ┤ √Xdg ├──■─────────────■──┤ √X ├
        #      ├──────┤┌─┴─┐┌───────┐┌─┴─┐├────┤
        # q_1: ┤ √Xdg ├┤ X ├┤ Rz(θ) ├┤ X ├┤ √X ├
        #      └──────┘└───┘└───────┘└───┘└────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.RYY._get_definition(self.params), add_regs=True, name=self.name
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-YY gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters, in which case it cannot
                yet be synthesized.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if annotated is None:
            annotated = any(isinstance(p, ParameterExpression) for p in self.params)

        gate = super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            annotated=annotated,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse RYY gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RYYGate` with an inverted parameter value.

        Returns:
            RYYGate: inverse gate.
        """
        return RYYGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the RYY gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta = float(self.params[0])
        cos = math.cos(theta / 2)
        isin = 1j * math.sin(theta / 2)
        return np.array(
            [[cos, 0, 0, isin], [0, cos, -isin, 0], [0, -isin, cos, 0], [isin, 0, 0, cos]],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RYYGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RYYGate):
            return self._compare_parameters(other)
        return False
