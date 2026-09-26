// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use smallvec::SmallVec;
use std::error;
use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{CustomOperation, Operation, Param};
use qiskit_circuit::py_convertible::PyConvertible;
use qiskit_synthesis::qft::qft_decompose_full::synth_qft_full;
use qiskit_util::py::ImportOnceCell;

/// The Python-space `QFTGate`, which holds a [`PyQftGate`] instance in its `_inner` attribute.
static QFT_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library.basis_change.qft", "QFTGate");

/// The Quantum Fourier Transform Gate.
///
/// On `n` qubits this is the operation
///
/// ```text
/// |j> -> 1/sqrt(2^n) * sum_k exp(2 pi i j k / 2^n) |k>
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFTGate {
    num_qubits: u32,
}

impl QFTGate {
    pub fn new(num_qubits: u32) -> Self {
        Self { num_qubits }
    }

    /// The number of qubits the QFT acts on.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }
}

impl Operation for QFTGate {
    fn name(&self) -> &str {
        "qft"
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for QFTGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn matrix(
        &self,
        _params: &[Param],
    ) -> Result<Option<Array2<Complex64>>, Box<dyn error::Error>> {
        // ToDo: should we return `None` if the number of qubits is too large?
        // This would also prevent overflow errors when computing 1 << num_qubits.
        let size = 1usize << self.num_qubits;
        let norm = (size as f64).sqrt().recip();
        Ok(Some(Array2::from_shape_fn((size, size), |(i, j)| {
            let phase = 2.0 * PI * (i * j) as f64 / (size as f64);
            Complex64::from_polar(norm, phase)
        })))
    }

    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        // Matches the Python `QFTGate._define`, which calls `synth_qft_full` with only
        // `num_qubits` set, leaving `do_swaps`, `approximation_degree` and `insert_barriers` at
        // their defaults.
        synth_qft_full(self.num_qubits as usize, true, 0, false)
            .ok()
            .map(CircuitData::from)
    }
}

/// Python-exposed wrapper around [`QFTGate`].
///
/// The Python-level ``qiskit.circuit.library.QFTGate`` (a :class:`~qiskit.circuit.Gate`
/// subclass) holds an instance of this class in a private ``_inner`` attribute, rather than
/// inheriting from it, and delegates to it for the pieces that are cheapest to implement once in
/// Rust: the dense matrix (via `__array__`/`matrix`). Equality, hashing, `repr`, copying and
/// pickling of the outer `QFTGate` are handled by the usual `Gate`/`Instruction` machinery in
/// Python instead of being inherited from here.
#[pyclass(
    module = "qiskit._accelerate.circuit_library",
    name = "QFTGate",
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyQftGate {
    inner: QFTGate,
}

#[pymethods]
impl PyQftGate {
    #[new]
    fn py_new(num_qubits: u32) -> Self {
        Self {
            inner: QFTGate::new(num_qubits),
        }
    }

    #[getter]
    #[pyo3(name = "num_qubits")]
    fn py_num_qubits(&self) -> u32 {
        self.inner.num_qubits
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.num_qubits.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("QFTGate({})", self.inner.num_qubits)
    }

    /// Support `copy.copy`. The gate is logically immutable, so returning the same instance
    /// is safe.
    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Support `copy.deepcopy`. There is no interior mutability or nested Python state, so
    /// sharing the instance is a complete deep copy; see `__copy__`.
    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(slf: Py<Self>, _memo: Option<&Bound<PyAny>>) -> Py<Self> {
        slf
    }

    /// Support `pickle`. This class is held by the public `qiskit.circuit.library.QFTGate` in a
    /// private `_inner` attribute rather than pickled directly as part of a circuit, but is
    /// still made picklable in its own right for convenience and consistency with other native
    /// types.
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (py.get_type::<Self>(), (self.inner.num_qubits,)).into_py_any(py)
    }

    /// The dense unitary matrix of this QFT, as a NumPy array.
    #[pyo3(name = "matrix")]
    fn py_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let matrix = CustomOperation::matrix(&self.inner, &[])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err("QFTGate has no matrix representation"))?;
        Ok(matrix.into_pyarray(py))
    }

    /// Support the NumPy array protocol (e.g. `np.asarray(gate)`), delegated to by the Python
    /// `QFTGate.__array__`. This always allocates a fresh array, so `copy=False` is rejected
    /// rather than honored.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "unable to avoid copy while creating an array as requested",
            ));
        }
        let array = self.py_matrix(py)?;
        let base_dtype = array.dtype();
        let dtype = dtype
            .map(|dtype| PyArrayDescr::new(py, dtype))
            .unwrap_or_else(|| Ok(base_dtype.clone()))?;
        if dtype.is_equiv_to(&base_dtype) {
            return Ok(array.into_any());
        }
        PyModule::import(py, intern!(py, "numpy"))?
            .getattr(intern!(py, "array"))?
            .call(
                (array,),
                Some(&[(intern!(py, "dtype"), dtype.as_any())].into_py_dict(py)?),
            )
    }

    fn definition(&self) -> PyResult<PyCircuitData> {
        let defn = self
            .inner
            .definition(&[])
            .expect("QFT should have definition");

        Ok(PyCircuitData { inner: defn })
    }
}

impl PyQftGate {
    pub fn new(inner: QFTGate) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &QFTGate {
        &self.inner
    }
}

/// The Python boundary for [`QFTGate`], kept out of [`CustomOperation`] so that the operation
/// itself stays independent of Python.
///
/// Serialization is not handled here: QPY constructs [`QFTGate`] directly, alongside the other
/// standard-library gates it knows.
impl PyConvertible for QFTGate {
    /// Wrap `self` in a [`PyQftGate`] and hand it to the Python `QFTGate._from_inner`
    /// classmethod, which builds a `QFTGate` around it without going through `__init__`.
    /// `QFTGate` in Python has no label and no params, so both arguments are ignored.
    fn create_py_op(
        &self,
        py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
        _label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let inner = PyQftGate::new(*self);
        Ok(QFT_GATE
            .get_bound(py)
            .call_method1(intern!(py, "_from_inner"), (inner,))?
            .unbind())
    }

    /// Returns `Ok(None)` if the object is not exactly a Python `QFTGate` (including if it is a
    /// *subclass* of `QFTGate`), so that the caller falls back to treating it as an opaque Python
    /// instruction rather than rejecting it.
    fn extract_from_py(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>> {
        if !ob.get_type().is(QFT_GATE.get_bound(ob.py())) {
            return Ok(None);
        }
        let Ok(inner) = ob.getattr(intern!(ob.py(), "_inner")) else {
            return Ok(None);
        };
        Ok(inner.extract::<PyQftGate>().ok().map(|gate| *gate.inner()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::OperationRef;
    use qiskit_circuit::packed_instruction::PackedOperation;

    // Tests basic QFT gate properties
    #[test]
    fn test_qft() {
        let qft3 = QFTGate::new(3);
        let another_qft3 = QFTGate::new(3);
        assert_eq!(qft3, another_qft3);

        let qft4 = QFTGate::new(4);
        assert_ne!(qft3, qft4);

        let mat = CustomOperation::matrix(&qft3, &[]);
        assert!(matches!(mat, Ok(Some(_))));
    }

    // Tests that the gate's definition matches the textbook synthesis it delegates to.
    #[test]
    fn test_qft_definition() {
        let qft = QFTGate::new(3);
        let definition =
            CustomOperation::definition(&qft, &[]).expect("QFTGate should have a definition");
        let expected =
            CircuitData::from(synth_qft_full(3, true, 0, false).expect("synthesis should succeed"));
        assert_eq!(definition.num_qubits(), expected.num_qubits());
        assert_eq!(definition.data().len(), expected.data().len());
    }

    // Tests putting QFT gates in a circuit and retrieving them back
    #[test]
    fn test_qft_rountrip() {
        let qft = QFTGate::new(4);

        // Add a QFT gate to a circuit
        let mut qc = CircuitData::with_capacity(1, 0, 1, 0.0.into())
            .expect("Circuit with small capacity should be built.");
        let qft_op = PackedOperation::from_custom_operation(Box::new(qft));
        qc.push_packed_operation(qft_op, None, &[Qubit(0)], &[])
            .expect("Instruction should be added to the circuit.");

        let retrieved_op = &qc.data()[0];

        let OperationRef::CustomOperation(dyn_cast_op) = retrieved_op.op.view() else {
            panic!("Gate should be a custom operation");
        };

        let Some(downcast_op) = dyn_cast_op.downcast_ref::<QFTGate>() else {
            panic!("Gate should be a custom gate of type QFTGate");
        };

        assert!(downcast_op.is_unitary());
        assert_eq!(downcast_op.num_qubits(), 4);
        assert_eq!(downcast_op, &qft);
    }

    /// Registration wires up both directions from the one `PyConvertible` impl.
    #[test]
    fn test_python_conversion_registered() {
        // Ignore the result: another test in this binary may have registered already, and this
        // asserts on the tables' contents rather than on which call populated them.
        let _ = crate::custom_operations::register_custom_operations();

        // Python -> Rust, keyed by name, which `NAME` keeps in step with `Operation::name`.
        assert_eq!("qft", Operation::name(&QFTGate::new(3)));
        assert!(qiskit_circuit::py_convertible::get_extractor("qft").is_some());
    }
}
