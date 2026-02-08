//! PyO3 Python Bindings for ALICE-ML
//!
//! "Pythonなのにカリカリ" — Python with Crispy Performance
//!
//! - GIL Release: Heavy computation runs GIL-free
//! - Zero-Copy NumPy: No data copying across FFI boundary
//! - Batch API: Push loops into Rust (SIMD + Rayon)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::ops::{ternary_matvec, ternary_matmul_batch, ternary_matvec_kernel};
use crate::ops::{TernaryWeight, TernaryWeightKernel};
use crate::quantize::{
    compute_quantization_error, dequantize_from_ternary, quantize_to_ternary,
    quantize_to_ternary_sparse, QuantStats,
};

// ============================================================================
// TernaryWeight
// ============================================================================

/// Packed ternary weight matrix (2-bit, 16x compression vs FP32)
#[pyclass(name = "TernaryWeight")]
pub struct PyTernaryWeight {
    pub(crate) inner: TernaryWeight,
}

#[pymethods]
impl PyTernaryWeight {
    /// Create from ternary values (-1, 0, +1) as int8 numpy array.
    #[new]
    fn new(values: PyReadonlyArray1<'_, i8>, out_features: usize, in_features: usize) -> PyResult<Self> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if slice.len() != out_features * in_features {
            return Err(PyValueError::new_err(format!(
                "values length {} != out_features({}) * in_features({})",
                slice.len(),
                out_features,
                in_features
            )));
        }
        Ok(Self {
            inner: TernaryWeight::from_ternary(slice, out_features, in_features),
        })
    }

    /// Create from pre-packed bytes with scale factor.
    #[staticmethod]
    fn from_packed(
        packed: Vec<u8>,
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Self {
        Self {
            inner: TernaryWeight::from_packed(packed, out_features, in_features, scale),
        }
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale()
    }

    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    /// Matrix-vector multiply: y = W * x (GIL released, SIMD)
    ///
    /// Input shape: (in_features,)
    /// Output shape: (out_features,)
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if input_slice.len() != self.inner.in_features() {
            return Err(PyValueError::new_err(format!(
                "input length {} != in_features {}",
                input_slice.len(),
                self.inner.in_features()
            )));
        }

        let out_features = self.inner.out_features();
        let weights = &self.inner;

        let result = py.detach(|| {
            let mut output = vec![0.0f32; out_features];
            ternary_matvec(input_slice, weights, &mut output);
            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Batched matrix multiply: Y = X @ W^T (GIL released, Rayon + SIMD)
    ///
    /// Input shape: (batch_size, in_features)
    /// Output shape: (batch_size, out_features)
    fn matmul_batch<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let array = input.as_array();
        let shape = array.shape();
        let batch_size = shape[0];
        let in_feat = shape[1];

        if in_feat != self.inner.in_features() {
            return Err(PyValueError::new_err(format!(
                "input features {} != weight in_features {}",
                in_feat,
                self.inner.in_features()
            )));
        }

        let input_slice = input.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out_features = self.inner.out_features();
        let weights = &self.inner;

        let result = py.detach(|| {
            let mut output = vec![0.0f32; batch_size * out_features];
            ternary_matmul_batch(input_slice, weights, &mut output, batch_size);
            output
        });

        let arr2d = Array2::from_shape_vec((batch_size, out_features), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr2d.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "TernaryWeight(out={}, in={}, scale={:.4}, mem={}B, ratio={:.1}x)",
            self.inner.out_features(),
            self.inner.in_features(),
            self.inner.scale(),
            self.inner.memory_bytes(),
            self.inner.compression_ratio()
        )
    }
}

// ============================================================================
// TernaryWeightKernel (bit-parallel SIMD)
// ============================================================================

/// Bit-parallel ternary weight kernel (optimized for SIMD, 32 weights per u32)
#[pyclass(name = "TernaryWeightKernel")]
pub struct PyTernaryWeightKernel {
    inner: TernaryWeightKernel,
}

#[pymethods]
impl PyTernaryWeightKernel {
    /// Create from ternary values (-1, 0, +1).
    #[new]
    fn new(values: PyReadonlyArray1<'_, i8>, out_features: usize, in_features: usize) -> PyResult<Self> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: TernaryWeightKernel::from_ternary(slice, out_features, in_features),
        })
    }

    /// Upgrade from TernaryWeight for faster SIMD execution.
    #[staticmethod]
    fn from_weight(weight: &PyTernaryWeight) -> Self {
        Self {
            inner: TernaryWeightKernel::from_packed_weight(&weight.inner),
        }
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    /// Matrix-vector multiply with bit-parallel kernel (GIL released).
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if input_slice.len() != self.inner.in_features() {
            return Err(PyValueError::new_err(format!(
                "input length {} != in_features {}",
                input_slice.len(),
                self.inner.in_features()
            )));
        }

        let out_features = self.inner.out_features();
        let kernel = &self.inner;

        let result = py.detach(|| {
            let mut output = vec![0.0f32; out_features];
            ternary_matvec_kernel(input_slice, kernel, &mut output);
            output
        });

        Ok(result.into_pyarray(py))
    }
}

// ============================================================================
// QuantStats
// ============================================================================

/// Quantization statistics after FP32 → ternary conversion.
#[pyclass(name = "QuantStats")]
#[derive(Clone)]
pub struct PyQuantStats {
    inner: QuantStats,
}

#[pymethods]
impl PyQuantStats {
    #[getter]
    fn plus_count(&self) -> usize {
        self.inner.plus_count
    }
    #[getter]
    fn minus_count(&self) -> usize {
        self.inner.minus_count
    }
    #[getter]
    fn zero_count(&self) -> usize {
        self.inner.zero_count
    }
    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }
    #[getter]
    fn mae(&self) -> f32 {
        self.inner.mae
    }
    #[getter]
    fn sparsity(&self) -> f32 {
        self.inner.sparsity()
    }
    #[getter]
    fn effective_bits(&self) -> f32 {
        self.inner.effective_bits()
    }
    #[getter]
    fn original_range(&self) -> (f32, f32) {
        self.inner.original_range
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantStats(+1={}, -1={}, 0={}, scale={:.4}, mae={:.6}, sparsity={:.1}%)",
            self.inner.plus_count,
            self.inner.minus_count,
            self.inner.zero_count,
            self.inner.scale,
            self.inner.mae,
            self.inner.sparsity() * 100.0
        )
    }
}

// ============================================================================
// Element-wise Operations (slice-based, GIL released)
// ============================================================================

/// Element-wise addition: c = a + b
#[pyfunction]
fn add<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sa = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sb = b.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if sa.len() != sb.len() {
        return Err(PyValueError::new_err("array length mismatch"));
    }
    let result = py.detach(|| {
        sa.iter().zip(sb.iter()).map(|(x, y)| x + y).collect::<Vec<f32>>()
    });
    Ok(result.into_pyarray(py))
}

/// Element-wise subtraction: c = a - b
#[pyfunction]
fn sub<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sa = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sb = b.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if sa.len() != sb.len() {
        return Err(PyValueError::new_err("array length mismatch"));
    }
    let result = py.detach(|| {
        sa.iter().zip(sb.iter()).map(|(x, y)| x - y).collect::<Vec<f32>>()
    });
    Ok(result.into_pyarray(py))
}

/// Scalar multiplication: c = a * scalar
#[pyfunction]
fn scale<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
    scalar: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sa = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = py.detach(|| sa.iter().map(|x| x * scalar).collect::<Vec<f32>>());
    Ok(result.into_pyarray(py))
}

/// ReLU activation: c = max(a, 0)
#[pyfunction]
fn relu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sa = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = py.detach(|| sa.iter().map(|&x| f32::max(x, 0.0)).collect::<Vec<f32>>());
    Ok(result.into_pyarray(py))
}

/// Softmax: exp(a_i - max) / sum(exp(a - max))
#[pyfunction]
fn softmax<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sa = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = py.detach(|| {
        let max_val = sa.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = sa.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / sa.len() as f32; sa.len()]
        }
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn sum(a: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let s = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(s.iter().sum())
}

#[pyfunction]
fn mean(a: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let s = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if s.is_empty() {
        return Ok(0.0);
    }
    Ok(s.iter().sum::<f32>() / s.len() as f32)
}

#[pyfunction]
fn max(a: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let s = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(s.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
}

#[pyfunction]
fn min(a: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let s = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(s.iter().cloned().fold(f32::INFINITY, f32::min))
}

// ============================================================================
// Quantization
// ============================================================================

/// Quantize FP32 weights to ternary {-1, 0, +1} (GIL released).
///
/// Uses BitNet b1.58 quantization method.
#[pyfunction]
fn quantize<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f32>,
    out_features: usize,
    in_features: usize,
) -> PyResult<(PyTernaryWeight, PyQuantStats)> {
    let w = weights.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if w.len() != out_features * in_features {
        return Err(PyValueError::new_err(format!(
            "weights length {} != out_features({}) * in_features({})",
            w.len(),
            out_features,
            in_features
        )));
    }

    let (tw, stats) = py.detach(|| quantize_to_ternary(w, out_features, in_features));

    Ok((PyTernaryWeight { inner: tw }, PyQuantStats { inner: stats }))
}

/// Quantize with sparsity threshold (GIL released).
#[pyfunction]
fn quantize_sparse<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f32>,
    out_features: usize,
    in_features: usize,
    threshold: f32,
) -> PyResult<(PyTernaryWeight, PyQuantStats)> {
    let w = weights.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if w.len() != out_features * in_features {
        return Err(PyValueError::new_err("weights length mismatch"));
    }

    let (tw, stats) =
        py.detach(|| quantize_to_ternary_sparse(w, out_features, in_features, threshold));

    Ok((PyTernaryWeight { inner: tw }, PyQuantStats { inner: stats }))
}

/// Dequantize ternary weights back to FP32.
#[pyfunction]
fn dequantize<'py>(
    py: Python<'py>,
    weights: &PyTernaryWeight,
) -> Bound<'py, PyArray1<f32>> {
    dequantize_from_ternary(&weights.inner).into_pyarray(py)
}

/// Compute quantization error metrics.
#[pyfunction]
fn quantization_error(
    original: PyReadonlyArray1<'_, f32>,
    quantized: &PyTernaryWeight,
) -> PyResult<(f32, f32, f32)> {
    let orig = original.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let err = compute_quantization_error(orig, &quantized.inner);
    Ok((err.mae, err.mse, err.max_error))
}

// ============================================================================
// Module
// ============================================================================

#[pymodule]
pub fn alice_ml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyTernaryWeight>()?;
    m.add_class::<PyTernaryWeightKernel>()?;
    m.add_class::<PyQuantStats>()?;

    // Element-wise ops
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;

    // Quantization
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantization_error, m)?)?;

    Ok(())
}
