//! C-ABI FFI for ALICE-ML
//!
//! 51 `extern "C"` functions with `am_ml_*` prefix.
//!
//! - Arena (7): arena lifecycle + alloc
//! - TernaryWeight (8): packed 2-bit weights
//! - TernaryWeightKernel (9): bit-parallel SIMD weights
//! - Matvec DPS (4): core ternary kernels
//! - Tensor DPS ops (13): element-wise ops
//! - BitLinear (5): neural layer
//! - Quantize (4): FP32 → ternary
//! - Version (1)
//!
//! Author: Moroya Sakamoto

use core::ffi::{c_char, c_int};
use std::sync::OnceLock;

use crate::arena::Arena;
use crate::layer::BitLinear;
use crate::ops::{
    ternary_matmul_batch, ternary_matvec, ternary_matvec_kernel, ternary_matvec_simd_dispatch,
    TernaryWeight, TernaryWeightKernel,
};
use crate::quantize::{compute_quantization_error, dequantize_from_ternary, quantize_to_ternary};
use crate::tensor::{
    tensor_add, tensor_copy, tensor_layer_norm, tensor_max, tensor_mean, tensor_min, tensor_relu,
    tensor_relu_inplace, tensor_rms_norm, tensor_scale, tensor_softmax, tensor_sub, tensor_sum,
};

// ============================================================================
// Arena (7)
// ============================================================================

/// 指定容量（バイト）のArenaを作成
#[no_mangle]
pub extern "C" fn am_ml_arena_new(capacity: usize) -> *mut Arena {
    Box::into_raw(Box::new(Arena::new(capacity)))
}

/// Arenaを解放
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_free(arena: *mut Arena) {
    if !arena.is_null() {
        drop(unsafe { Box::from_raw(arena) });
    }
}

/// Arenaをリセット（オフセットをゼロに戻す）
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_reset(arena: *mut Arena) {
    if let Some(a) = unsafe { arena.as_mut() } {
        a.reset();
    }
}

/// Arena使用量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_used(arena: *const Arena) -> usize {
    unsafe { arena.as_ref() }.map_or(0, |a| a.used())
}

/// Arena容量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_capacity(arena: *const Arena) -> usize {
    unsafe { arena.as_ref() }.map_or(0, |a| a.capacity())
}

/// Arena残容量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_remaining(arena: *const Arena) -> usize {
    unsafe { arena.as_ref() }.map_or(0, |a| a.remaining())
}

/// Arenaからf32配列を確保。失敗時はnullを返す。
/// 返されたポインタはArenaのリセット・解放後は無効。
#[no_mangle]
pub unsafe extern "C" fn am_ml_arena_alloc_f32(
    arena: *mut Arena,
    count: usize,
) -> *mut f32 {
    let Some(a) = (unsafe { arena.as_mut() }) else {
        return core::ptr::null_mut();
    };
    match a.alloc::<f32>(count) {
        Some(slice) => slice.as_mut_ptr(),
        None => core::ptr::null_mut(),
    }
}

// ============================================================================
// TernaryWeight (8)
// ============================================================================

/// 三値重みを作成（values: i8配列 {-1,0,+1}、len = out_features * in_features）
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_from_ternary(
    values: *const i8,
    len: usize,
    out_features: usize,
    in_features: usize,
) -> *mut TernaryWeight {
    if values.is_null() || len == 0 {
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(values, len) };
    let w = TernaryWeight::from_ternary(slice, out_features, in_features);
    Box::into_raw(Box::new(w))
}

/// TernaryWeightを解放
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_free(w: *mut TernaryWeight) {
    if !w.is_null() {
        drop(unsafe { Box::from_raw(w) });
    }
}

/// 出力特徴数
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_out_features(w: *const TernaryWeight) -> usize {
    unsafe { w.as_ref() }.map_or(0, |w| w.out_features())
}

/// 入力特徴数
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_in_features(w: *const TernaryWeight) -> usize {
    unsafe { w.as_ref() }.map_or(0, |w| w.in_features())
}

/// スケール係数
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_scale(w: *const TernaryWeight) -> f32 {
    unsafe { w.as_ref() }.map_or(0.0, |w| w.scale())
}

/// VRAM使用量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_memory_bytes(w: *const TernaryWeight) -> usize {
    unsafe { w.as_ref() }.map_or(0, |w| w.memory_bytes())
}

/// FP32比の圧縮率
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_compression_ratio(w: *const TernaryWeight) -> f32 {
    unsafe { w.as_ref() }.map_or(0.0, |w| w.compression_ratio())
}

/// 指定位置の重みを取得（i8: -1, 0, +1）
#[no_mangle]
pub unsafe extern "C" fn am_ml_weight_get(
    w: *const TernaryWeight,
    row: usize,
    col: usize,
) -> i8 {
    unsafe { w.as_ref() }.map_or(0, |w| w.get(row, col).to_i8())
}

// ============================================================================
// TernaryWeightKernel (9)
// ============================================================================

/// ビットパラレル重みを作成（スケール1.0）
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_from_ternary(
    values: *const i8,
    len: usize,
    out_features: usize,
    in_features: usize,
) -> *mut TernaryWeightKernel {
    if values.is_null() || len == 0 {
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(values, len) };
    let k = TernaryWeightKernel::from_ternary(slice, out_features, in_features);
    Box::into_raw(Box::new(k))
}

/// スケール付きビットパラレル重みを作成
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_from_ternary_scaled(
    values: *const i8,
    len: usize,
    out_features: usize,
    in_features: usize,
    scale: f32,
) -> *mut TernaryWeightKernel {
    if values.is_null() || len == 0 {
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(values, len) };
    let k = TernaryWeightKernel::from_ternary_scaled(slice, out_features, in_features, scale);
    Box::into_raw(Box::new(k))
}

/// TernaryWeightからビットパラレル形式に変換
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_from_weight(
    w: *const TernaryWeight,
) -> *mut TernaryWeightKernel {
    let Some(w) = (unsafe { w.as_ref() }) else {
        return core::ptr::null_mut();
    };
    let k = TernaryWeightKernel::from_packed_weight(w);
    Box::into_raw(Box::new(k))
}

/// TernaryWeightKernelを解放
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_free(k: *mut TernaryWeightKernel) {
    if !k.is_null() {
        drop(unsafe { Box::from_raw(k) });
    }
}

/// 出力特徴数
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_out_features(k: *const TernaryWeightKernel) -> usize {
    unsafe { k.as_ref() }.map_or(0, |k| k.out_features())
}

/// 入力特徴数
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_in_features(k: *const TernaryWeightKernel) -> usize {
    unsafe { k.as_ref() }.map_or(0, |k| k.in_features())
}

/// メモリ使用量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_memory_bytes(k: *const TernaryWeightKernel) -> usize {
    unsafe { k.as_ref() }.map_or(0, |k| k.memory_bytes())
}

/// FP32比の圧縮率
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_compression_ratio(k: *const TernaryWeightKernel) -> f32 {
    unsafe { k.as_ref() }.map_or(0.0, |k| k.compression_ratio())
}

/// 行あたりのu32ワード数
#[no_mangle]
pub unsafe extern "C" fn am_ml_kernel_words_per_row(k: *const TernaryWeightKernel) -> usize {
    unsafe { k.as_ref() }.map_or(0, |k| k.words_per_row())
}

// ============================================================================
// Matvec DPS (4)
// ============================================================================

/// 三値行列-ベクトル積（パック重み版）
/// output[0..out_len] = W * input[0..in_len]
#[no_mangle]
pub unsafe extern "C" fn am_ml_matvec(
    input: *const f32,
    in_len: usize,
    weights: *const TernaryWeight,
    output: *mut f32,
    out_len: usize,
) {
    let (Some(w), false, false) = (
        unsafe { weights.as_ref() },
        input.is_null(),
        output.is_null(),
    ) else {
        return;
    };
    let inp = unsafe { core::slice::from_raw_parts(input, in_len) };
    let out = unsafe { core::slice::from_raw_parts_mut(output, out_len) };
    ternary_matvec(inp, w, out);
}

/// 三値行列-ベクトル積（ビットパラレルカーネル版）
#[no_mangle]
pub unsafe extern "C" fn am_ml_matvec_kernel(
    input: *const f32,
    in_len: usize,
    kernel: *const TernaryWeightKernel,
    output: *mut f32,
    out_len: usize,
) {
    let (Some(k), false, false) = (
        unsafe { kernel.as_ref() },
        input.is_null(),
        output.is_null(),
    ) else {
        return;
    };
    let inp = unsafe { core::slice::from_raw_parts(input, in_len) };
    let out = unsafe { core::slice::from_raw_parts_mut(output, out_len) };
    ternary_matvec_kernel(inp, k, out);
}

/// バッチ行列乗算（パック重み版）
/// input: batch_size × in_features, output: batch_size × out_features
#[no_mangle]
pub unsafe extern "C" fn am_ml_matmul_batch(
    input: *const f32,
    in_len: usize,
    weights: *const TernaryWeight,
    output: *mut f32,
    out_len: usize,
    batch_size: usize,
) {
    let (Some(w), false, false) = (
        unsafe { weights.as_ref() },
        input.is_null(),
        output.is_null(),
    ) else {
        return;
    };
    let inp = unsafe { core::slice::from_raw_parts(input, in_len) };
    let out = unsafe { core::slice::from_raw_parts_mut(output, out_len) };
    ternary_matmul_batch(inp, w, out, batch_size);
}

/// SIMD自動ディスパッチ（AVX2/NEON/スカラー）
#[no_mangle]
pub unsafe extern "C" fn am_ml_matvec_simd(
    input: *const f32,
    in_len: usize,
    kernel: *const TernaryWeightKernel,
    output: *mut f32,
    out_len: usize,
) {
    let (Some(k), false, false) = (
        unsafe { kernel.as_ref() },
        input.is_null(),
        output.is_null(),
    ) else {
        return;
    };
    let inp = unsafe { core::slice::from_raw_parts(input, in_len) };
    let out = unsafe { core::slice::from_raw_parts_mut(output, out_len) };
    ternary_matvec_simd_dispatch(inp, k, out);
}

// ============================================================================
// Tensor DPS ops (13)
// ============================================================================

/// element-wise 加算: out[i] = a[i] + b[i]
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_add(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
) {
    if a.is_null() || b.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let sb = unsafe { core::slice::from_raw_parts(b, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_add(sa, sb, so);
}

/// element-wise 減算: out[i] = a[i] - b[i]
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_sub(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
) {
    if a.is_null() || b.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let sb = unsafe { core::slice::from_raw_parts(b, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_sub(sa, sb, so);
}

/// スカラー乗算: out[i] = a[i] * s
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_scale(
    a: *const f32,
    s: f32,
    out: *mut f32,
    len: usize,
) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_scale(sa, s, so);
}

/// コピー: out[i] = a[i]
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_copy(a: *const f32, out: *mut f32, len: usize) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_copy(sa, so);
}

/// 合計
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_sum(a: *const f32, len: usize) -> f32 {
    if a.is_null() {
        return 0.0;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    tensor_sum(sa)
}

/// 平均
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_mean(a: *const f32, len: usize) -> f32 {
    if a.is_null() {
        return 0.0;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    tensor_mean(sa)
}

/// 最小値
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_min(a: *const f32, len: usize) -> f32 {
    if a.is_null() {
        return 0.0;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    tensor_min(sa)
}

/// 最大値
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_max(a: *const f32, len: usize) -> f32 {
    if a.is_null() {
        return 0.0;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    tensor_max(sa)
}

/// ReLU: out[i] = max(0, a[i])
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_relu(a: *const f32, out: *mut f32, len: usize) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_relu(sa, so);
}

/// ReLUインプレース: a[i] = max(0, a[i])
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_relu_inplace(a: *mut f32, len: usize) {
    if a.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts_mut(a, len) };
    tensor_relu_inplace(sa);
}

/// Softmax: out = softmax(a)
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_softmax(a: *const f32, out: *mut f32, len: usize) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_softmax(sa, so);
}

/// RMSNorm: out = rms_norm(a, epsilon)
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_rms_norm(
    a: *const f32,
    epsilon: f32,
    out: *mut f32,
    len: usize,
) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_rms_norm(sa, epsilon, so);
}

/// LayerNorm: out = layer_norm(a, epsilon)
#[no_mangle]
pub unsafe extern "C" fn am_ml_tensor_layer_norm(
    a: *const f32,
    epsilon: f32,
    out: *mut f32,
    len: usize,
) {
    if a.is_null() || out.is_null() {
        return;
    }
    let sa = unsafe { core::slice::from_raw_parts(a, len) };
    let so = unsafe { core::slice::from_raw_parts_mut(out, len) };
    tensor_layer_norm(sa, epsilon, so);
}

// ============================================================================
// BitLinear (5)
// ============================================================================

/// BitLinearレイヤーを作成
/// kernel: 所有権を移譲（呼び出し後にポインタを使用しないこと）
/// bias: nullの場合はバイアスなし
/// pre_norm: 0=false, 1=true
#[no_mangle]
pub unsafe extern "C" fn am_ml_bitlinear_new(
    kernel: *mut TernaryWeightKernel,
    bias: *const f32,
    bias_len: usize,
    pre_norm: c_int,
) -> *mut BitLinear {
    if kernel.is_null() {
        return core::ptr::null_mut();
    }
    let k = unsafe { *Box::from_raw(kernel) };
    let b = if bias.is_null() || bias_len == 0 {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(bias, bias_len) }.to_vec())
    };
    let layer = BitLinear::new(k, b, pre_norm != 0);
    Box::into_raw(Box::new(layer))
}

/// BitLinearを解放
#[no_mangle]
pub unsafe extern "C" fn am_ml_bitlinear_free(layer: *mut BitLinear) {
    if !layer.is_null() {
        drop(unsafe { Box::from_raw(layer) });
    }
}

/// フォワードパス（DPS）
#[no_mangle]
pub unsafe extern "C" fn am_ml_bitlinear_forward(
    layer: *const BitLinear,
    input: *const f32,
    in_len: usize,
    output: *mut f32,
    out_len: usize,
) {
    let (Some(l), false, false) = (
        unsafe { layer.as_ref() },
        input.is_null(),
        output.is_null(),
    ) else {
        return;
    };
    let inp = unsafe { core::slice::from_raw_parts(input, in_len) };
    let out = unsafe { core::slice::from_raw_parts_mut(output, out_len) };
    l.forward(inp, out);
}

/// レイヤーのメモリ使用量（バイト）
#[no_mangle]
pub unsafe extern "C" fn am_ml_bitlinear_memory_bytes(layer: *const BitLinear) -> usize {
    unsafe { layer.as_ref() }.map_or(0, |l| l.memory_bytes())
}

/// FP32比の圧縮率
#[no_mangle]
pub unsafe extern "C" fn am_ml_bitlinear_compression_ratio(layer: *const BitLinear) -> f32 {
    unsafe { layer.as_ref() }.map_or(0.0, |l| l.compression_ratio())
}

// ============================================================================
// Quantize (4)
// ============================================================================

/// FP32重みを三値量子化（BitNet b1.58方式）
/// 返り値はTernaryWeightポインタ（am_ml_weight_freeで解放）
#[no_mangle]
pub unsafe extern "C" fn am_ml_quantize(
    weights: *const f32,
    len: usize,
    out_features: usize,
    in_features: usize,
) -> *mut TernaryWeight {
    if weights.is_null() || len == 0 {
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(weights, len) };
    let (tw, _stats) = quantize_to_ternary(slice, out_features, in_features);
    Box::into_raw(Box::new(tw))
}

/// 三値重みをFP32に逆量子化
/// outに書き込んだ要素数を返す
#[no_mangle]
pub unsafe extern "C" fn am_ml_dequantize(
    w: *const TernaryWeight,
    out: *mut f32,
    max_len: usize,
) -> usize {
    let Some(w) = (unsafe { w.as_ref() }) else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let deq = dequantize_from_ternary(w);
    let n = deq.len().min(max_len);
    let dst = unsafe { core::slice::from_raw_parts_mut(out, n) };
    dst.copy_from_slice(&deq[..n]);
    n
}

/// 量子化誤差（MAE）
#[no_mangle]
pub unsafe extern "C" fn am_ml_quantization_error_mae(
    original: *const f32,
    len: usize,
    quantized: *const TernaryWeight,
) -> f32 {
    let (Some(q), false) = (unsafe { quantized.as_ref() }, original.is_null()) else {
        return -1.0;
    };
    let orig = unsafe { core::slice::from_raw_parts(original, len) };
    let err = compute_quantization_error(orig, q);
    err.mae
}

/// 量子化誤差（SNR dB）
#[no_mangle]
pub unsafe extern "C" fn am_ml_quantization_error_snr(
    original: *const f32,
    len: usize,
    quantized: *const TernaryWeight,
) -> f32 {
    let (Some(q), false) = (unsafe { quantized.as_ref() }, original.is_null()) else {
        return -1.0;
    };
    let orig = unsafe { core::slice::from_raw_parts(original, len) };
    let err = compute_quantization_error(orig, q);
    err.snr
}

// ============================================================================
// Version (1)
// ============================================================================

/// ライブラリバージョン文字列（静的寿命、解放不要）
#[no_mangle]
pub extern "C" fn am_ml_version() -> *const c_char {
    static VERSION_C: OnceLock<std::ffi::CString> = OnceLock::new();
    VERSION_C
        .get_or_init(|| std::ffi::CString::new(crate::VERSION).unwrap())
        .as_ptr()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = am_ml_version();
        assert!(!v.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(v) }.to_str().unwrap();
        assert!(s.starts_with("0."));
    }

    #[test]
    fn test_arena_lifecycle() {
        let arena = am_ml_arena_new(4096);
        assert!(!arena.is_null());
        assert_eq!(unsafe { am_ml_arena_capacity(arena) }, 4096);
        assert_eq!(unsafe { am_ml_arena_used(arena) }, 0);

        let ptr = unsafe { am_ml_arena_alloc_f32(arena, 10) };
        assert!(!ptr.is_null());
        assert!(unsafe { am_ml_arena_used(arena) } > 0);

        unsafe { am_ml_arena_reset(arena) };
        assert_eq!(unsafe { am_ml_arena_used(arena) }, 0);

        unsafe { am_ml_arena_free(arena) };
    }

    #[test]
    fn test_weight_lifecycle() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { am_ml_weight_from_ternary(values.as_ptr(), 4, 2, 2) };
        assert!(!w.is_null());

        assert_eq!(unsafe { am_ml_weight_out_features(w) }, 2);
        assert_eq!(unsafe { am_ml_weight_in_features(w) }, 2);
        assert!((unsafe { am_ml_weight_scale(w) } - 1.0).abs() < 1e-6);
        assert!(unsafe { am_ml_weight_memory_bytes(w) } > 0);
        assert!(unsafe { am_ml_weight_compression_ratio(w) } > 0.0);

        assert_eq!(unsafe { am_ml_weight_get(w, 0, 0) }, 1);
        assert_eq!(unsafe { am_ml_weight_get(w, 0, 1) }, -1);
        assert_eq!(unsafe { am_ml_weight_get(w, 1, 0) }, 0);
        assert_eq!(unsafe { am_ml_weight_get(w, 1, 1) }, 1);

        unsafe { am_ml_weight_free(w) };
    }

    #[test]
    fn test_kernel_lifecycle() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary(values.as_ptr(), 4, 2, 2) };
        assert!(!k.is_null());

        assert_eq!(unsafe { am_ml_kernel_out_features(k) }, 2);
        assert_eq!(unsafe { am_ml_kernel_in_features(k) }, 2);
        assert!(unsafe { am_ml_kernel_memory_bytes(k) } > 0);
        assert!(unsafe { am_ml_kernel_words_per_row(k) } > 0);

        unsafe { am_ml_kernel_free(k) };
    }

    #[test]
    fn test_kernel_from_weight() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { am_ml_weight_from_ternary(values.as_ptr(), 4, 2, 2) };
        let k = unsafe { am_ml_kernel_from_weight(w) };
        assert!(!k.is_null());
        assert_eq!(unsafe { am_ml_kernel_out_features(k) }, 2);

        unsafe { am_ml_kernel_free(k) };
        unsafe { am_ml_weight_free(w) };
    }

    #[test]
    fn test_kernel_scaled() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary_scaled(values.as_ptr(), 4, 2, 2, 2.5) };
        assert!(!k.is_null());
        assert!((unsafe { am_ml_kernel_compression_ratio(k) }) > 0.0);
        unsafe { am_ml_kernel_free(k) };
    }

    #[test]
    fn test_matvec() {
        // W = [[1,-1],[0,1]], x = [2,3]
        // y[0] = 2 - 3 = -1, y[1] = 0 + 3 = 3
        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { am_ml_weight_from_ternary(values.as_ptr(), 4, 2, 2) };

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        unsafe { am_ml_matvec(input.as_ptr(), 2, w, output.as_mut_ptr(), 2) };

        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);

        unsafe { am_ml_weight_free(w) };
    }

    #[test]
    fn test_matvec_kernel() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary(values.as_ptr(), 4, 2, 2) };

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        unsafe { am_ml_matvec_kernel(input.as_ptr(), 2, k, output.as_mut_ptr(), 2) };

        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);

        unsafe { am_ml_kernel_free(k) };
    }

    #[test]
    fn test_matmul_batch() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { am_ml_weight_from_ternary(values.as_ptr(), 4, 2, 2) };

        // batch_size=2: [2,3] and [1,1]
        let input = [2.0f32, 3.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        unsafe { am_ml_matmul_batch(input.as_ptr(), 4, w, output.as_mut_ptr(), 4, 2) };

        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);

        unsafe { am_ml_weight_free(w) };
    }

    #[test]
    fn test_matvec_simd() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary(values.as_ptr(), 4, 2, 2) };

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        unsafe { am_ml_matvec_simd(input.as_ptr(), 2, k, output.as_mut_ptr(), 2) };

        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);

        unsafe { am_ml_kernel_free(k) };
    }

    #[test]
    fn test_tensor_ops() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];
        let mut out = [0.0f32; 4];

        unsafe { am_ml_tensor_add(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) };
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[3] - 6.0).abs() < 1e-6);

        unsafe { am_ml_tensor_sub(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) };
        assert!((out[0] - 0.5).abs() < 1e-6);

        unsafe { am_ml_tensor_scale(a.as_ptr(), 2.0, out.as_mut_ptr(), 4) };
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[3] - 8.0).abs() < 1e-6);

        assert!((unsafe { am_ml_tensor_sum(a.as_ptr(), 4) } - 10.0).abs() < 1e-6);
        assert!((unsafe { am_ml_tensor_mean(a.as_ptr(), 4) } - 2.5).abs() < 1e-6);
        assert!((unsafe { am_ml_tensor_min(a.as_ptr(), 4) } - 1.0).abs() < 1e-6);
        assert!((unsafe { am_ml_tensor_max(a.as_ptr(), 4) } - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_relu() {
        let a = [-1.0f32, 0.0, 1.0, -2.0];
        let mut out = [0.0f32; 4];

        unsafe { am_ml_tensor_relu(a.as_ptr(), out.as_mut_ptr(), 4) };
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);

        let mut inplace = [-1.0f32, 2.0, -3.0, 4.0];
        unsafe { am_ml_tensor_relu_inplace(inplace.as_mut_ptr(), 4) };
        assert!((inplace[0] - 0.0).abs() < 1e-6);
        assert!((inplace[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_softmax() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        unsafe { am_ml_tensor_softmax(a.as_ptr(), out.as_mut_ptr(), 3) };
        let total: f32 = out.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_norms() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];

        unsafe { am_ml_tensor_rms_norm(a.as_ptr(), 1e-5, out.as_mut_ptr(), 4) };
        // RMS norm should produce unit-variance-ish output
        let rms_sum: f32 = out.iter().map(|x| x * x).sum();
        assert!((rms_sum / 4.0 - 1.0).abs() < 0.1);

        unsafe { am_ml_tensor_layer_norm(a.as_ptr(), 1e-5, out.as_mut_ptr(), 4) };
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_bitlinear() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary(values.as_ptr(), 4, 2, 2) };

        // バイアスなし、pre_norm=false
        let layer = unsafe { am_ml_bitlinear_new(k, core::ptr::null(), 0, 0) };
        assert!(!layer.is_null());

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        unsafe { am_ml_bitlinear_forward(layer, input.as_ptr(), 2, output.as_mut_ptr(), 2) };

        assert!((output[0] - (-1.0)).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);

        assert!(unsafe { am_ml_bitlinear_memory_bytes(layer) } > 0);
        assert!(unsafe { am_ml_bitlinear_compression_ratio(layer) } > 0.0);

        unsafe { am_ml_bitlinear_free(layer) };
    }

    #[test]
    fn test_bitlinear_with_bias() {
        let values: [i8; 4] = [1, -1, 0, 1];
        let k = unsafe { am_ml_kernel_from_ternary(values.as_ptr(), 4, 2, 2) };
        let bias = [10.0f32, -5.0];

        let layer = unsafe { am_ml_bitlinear_new(k, bias.as_ptr(), 2, 0) };

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        unsafe { am_ml_bitlinear_forward(layer, input.as_ptr(), 2, output.as_mut_ptr(), 2) };

        // y[0] = -1 + 10 = 9, y[1] = 3 - 5 = -2
        assert!((output[0] - 9.0).abs() < 1e-5);
        assert!((output[1] - (-2.0)).abs() < 1e-5);

        unsafe { am_ml_bitlinear_free(layer) };
    }

    #[test]
    fn test_quantize_roundtrip() {
        let fp32: [f32; 4] = [0.5, -0.3, 0.0, 0.8];
        let w = unsafe { am_ml_quantize(fp32.as_ptr(), 4, 2, 2) };
        assert!(!w.is_null());

        let mut deq = [0.0f32; 4];
        let n = unsafe { am_ml_dequantize(w, deq.as_mut_ptr(), 4) };
        assert_eq!(n, 4);

        let mae = unsafe { am_ml_quantization_error_mae(fp32.as_ptr(), 4, w) };
        assert!(mae >= 0.0);

        let snr = unsafe { am_ml_quantization_error_snr(fp32.as_ptr(), 4, w) };
        assert!(snr > 0.0);

        unsafe { am_ml_weight_free(w) };
    }

    #[test]
    fn test_null_safety() {
        // 全関数がnullポインタでクラッシュしないことを確認
        unsafe {
            am_ml_arena_free(core::ptr::null_mut());
            am_ml_arena_reset(core::ptr::null_mut());
            assert_eq!(am_ml_arena_used(core::ptr::null()), 0);
            assert_eq!(am_ml_arena_capacity(core::ptr::null()), 0);
            assert_eq!(am_ml_arena_remaining(core::ptr::null()), 0);
            assert!(am_ml_arena_alloc_f32(core::ptr::null_mut(), 10).is_null());

            am_ml_weight_free(core::ptr::null_mut());
            assert_eq!(am_ml_weight_out_features(core::ptr::null()), 0);
            assert_eq!(am_ml_weight_in_features(core::ptr::null()), 0);
            assert_eq!(am_ml_weight_get(core::ptr::null(), 0, 0), 0);

            am_ml_kernel_free(core::ptr::null_mut());
            assert_eq!(am_ml_kernel_out_features(core::ptr::null()), 0);
            assert!(am_ml_kernel_from_weight(core::ptr::null()).is_null());

            am_ml_matvec(core::ptr::null(), 0, core::ptr::null(), core::ptr::null_mut(), 0);
            am_ml_matvec_kernel(core::ptr::null(), 0, core::ptr::null(), core::ptr::null_mut(), 0);

            am_ml_tensor_add(
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                0,
            );
            assert_eq!(am_ml_tensor_sum(core::ptr::null(), 0), 0.0);

            am_ml_bitlinear_free(core::ptr::null_mut());
            assert!(am_ml_bitlinear_new(core::ptr::null_mut(), core::ptr::null(), 0, 0).is_null());
            assert_eq!(am_ml_bitlinear_memory_bytes(core::ptr::null()), 0);

            assert!(am_ml_quantize(core::ptr::null(), 0, 0, 0).is_null());
            assert_eq!(am_ml_dequantize(core::ptr::null(), core::ptr::null_mut(), 0), 0);
        }
    }

    #[test]
    fn test_tensor_copy_ffi() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        unsafe { am_ml_tensor_copy(a.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_arena_alloc_exhaustion() {
        let arena = am_ml_arena_new(32);
        let ptr = unsafe { am_ml_arena_alloc_f32(arena, 1000) };
        assert!(ptr.is_null());
        unsafe { am_ml_arena_free(arena) };
    }
}
