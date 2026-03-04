// ALICE-ML Unity C# Bindings
// 51 DllImport + 7 RAII IDisposable handles
//
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;

namespace Alice.ML
{
    // ========================================================================
    // Native P/Invoke (51 functions)
    // ========================================================================

    internal static class Native
    {
        const string DLL = "alice_ml";

        // ---- Arena (7) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_arena_new(UIntPtr capacity);
        [DllImport(DLL)] internal static extern void am_ml_arena_free(IntPtr arena);
        [DllImport(DLL)] internal static extern void am_ml_arena_reset(IntPtr arena);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_arena_used(IntPtr arena);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_arena_capacity(IntPtr arena);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_arena_remaining(IntPtr arena);
        [DllImport(DLL)] internal static extern IntPtr am_ml_arena_alloc_f32(IntPtr arena, UIntPtr count);

        // ---- TernaryWeight (8) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_weight_from_ternary(sbyte[] values, UIntPtr len, UIntPtr outFeatures, UIntPtr inFeatures);
        [DllImport(DLL)] internal static extern void am_ml_weight_free(IntPtr w);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_weight_out_features(IntPtr w);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_weight_in_features(IntPtr w);
        [DllImport(DLL)] internal static extern float am_ml_weight_scale(IntPtr w);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_weight_memory_bytes(IntPtr w);
        [DllImport(DLL)] internal static extern float am_ml_weight_compression_ratio(IntPtr w);
        [DllImport(DLL)] internal static extern sbyte am_ml_weight_get(IntPtr w, UIntPtr row, UIntPtr col);

        // ---- TernaryWeightKernel (9) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_kernel_from_ternary(sbyte[] values, UIntPtr len, UIntPtr outFeatures, UIntPtr inFeatures);
        [DllImport(DLL)] internal static extern IntPtr am_ml_kernel_from_ternary_scaled(sbyte[] values, UIntPtr len, UIntPtr outFeatures, UIntPtr inFeatures, float scale);
        [DllImport(DLL)] internal static extern IntPtr am_ml_kernel_from_weight(IntPtr w);
        [DllImport(DLL)] internal static extern void am_ml_kernel_free(IntPtr k);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_kernel_out_features(IntPtr k);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_kernel_in_features(IntPtr k);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_kernel_memory_bytes(IntPtr k);
        [DllImport(DLL)] internal static extern float am_ml_kernel_compression_ratio(IntPtr k);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_kernel_words_per_row(IntPtr k);

        // ---- Matvec DPS (4) ----
        [DllImport(DLL)] internal static extern void am_ml_matvec(float[] input, UIntPtr inLen, IntPtr weights, float[] output, UIntPtr outLen);
        [DllImport(DLL)] internal static extern void am_ml_matvec_kernel(float[] input, UIntPtr inLen, IntPtr kernel, float[] output, UIntPtr outLen);
        [DllImport(DLL)] internal static extern void am_ml_matmul_batch(float[] input, UIntPtr inLen, IntPtr weights, float[] output, UIntPtr outLen, UIntPtr batchSize);
        [DllImport(DLL)] internal static extern void am_ml_matvec_simd(float[] input, UIntPtr inLen, IntPtr kernel, float[] output, UIntPtr outLen);

        // ---- Tensor DPS ops (13) ----
        [DllImport(DLL)] internal static extern void am_ml_tensor_add(float[] a, float[] b, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_sub(float[] a, float[] b, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_scale(float[] a, float s, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_copy(float[] a, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern float am_ml_tensor_sum(float[] a, UIntPtr len);
        [DllImport(DLL)] internal static extern float am_ml_tensor_mean(float[] a, UIntPtr len);
        [DllImport(DLL)] internal static extern float am_ml_tensor_min(float[] a, UIntPtr len);
        [DllImport(DLL)] internal static extern float am_ml_tensor_max(float[] a, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_relu(float[] a, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_relu_inplace(float[] a, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_softmax(float[] a, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_rms_norm(float[] a, float epsilon, float[] outBuf, UIntPtr len);
        [DllImport(DLL)] internal static extern void am_ml_tensor_layer_norm(float[] a, float epsilon, float[] outBuf, UIntPtr len);

        // ---- BitLinear (5) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_bitlinear_new(IntPtr kernel, float[] bias, UIntPtr biasLen, int preNorm);
        [DllImport(DLL)] internal static extern void am_ml_bitlinear_free(IntPtr layer);
        [DllImport(DLL)] internal static extern void am_ml_bitlinear_forward(IntPtr layer, float[] input, UIntPtr inLen, float[] output, UIntPtr outLen);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_bitlinear_memory_bytes(IntPtr layer);
        [DllImport(DLL)] internal static extern float am_ml_bitlinear_compression_ratio(IntPtr layer);

        // ---- Quantize (4) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_quantize(float[] weights, UIntPtr len, UIntPtr outFeatures, UIntPtr inFeatures);
        [DllImport(DLL)] internal static extern UIntPtr am_ml_dequantize(IntPtr w, float[] outBuf, UIntPtr maxLen);
        [DllImport(DLL)] internal static extern float am_ml_quantization_error_mae(float[] original, UIntPtr len, IntPtr quantized);
        [DllImport(DLL)] internal static extern float am_ml_quantization_error_snr(float[] original, UIntPtr len, IntPtr quantized);

        // ---- Version (1) ----
        [DllImport(DLL)] internal static extern IntPtr am_ml_version();
    }

    // ========================================================================
    // RAII Handles
    // ========================================================================

    /// Arena bump allocator handle
    public sealed class ArenaHandle : IDisposable
    {
        internal IntPtr Ptr;
        public ArenaHandle(ulong capacity) { Ptr = Native.am_ml_arena_new((UIntPtr)capacity); }
        public void Reset() => Native.am_ml_arena_reset(Ptr);
        public ulong Used => (ulong)Native.am_ml_arena_used(Ptr);
        public ulong Capacity => (ulong)Native.am_ml_arena_capacity(Ptr);
        public ulong Remaining => (ulong)Native.am_ml_arena_remaining(Ptr);
        public void Dispose() { if (Ptr != IntPtr.Zero) { Native.am_ml_arena_free(Ptr); Ptr = IntPtr.Zero; } }
    }

    /// Packed ternary weight handle
    public sealed class TernaryWeightHandle : IDisposable
    {
        internal IntPtr Ptr;
        internal TernaryWeightHandle(IntPtr ptr) { Ptr = ptr; }

        public static TernaryWeightHandle FromTernary(sbyte[] values, int outFeatures, int inFeatures)
            => new TernaryWeightHandle(Native.am_ml_weight_from_ternary(values, (UIntPtr)values.Length, (UIntPtr)outFeatures, (UIntPtr)inFeatures));

        public int OutFeatures => (int)Native.am_ml_weight_out_features(Ptr);
        public int InFeatures => (int)Native.am_ml_weight_in_features(Ptr);
        public float Scale => Native.am_ml_weight_scale(Ptr);
        public int MemoryBytes => (int)Native.am_ml_weight_memory_bytes(Ptr);
        public float CompressionRatio => Native.am_ml_weight_compression_ratio(Ptr);
        public sbyte Get(int row, int col) => Native.am_ml_weight_get(Ptr, (UIntPtr)row, (UIntPtr)col);

        public void Matvec(float[] input, float[] output)
            => Native.am_ml_matvec(input, (UIntPtr)input.Length, Ptr, output, (UIntPtr)output.Length);

        public void MatmulBatch(float[] input, float[] output, int batchSize)
            => Native.am_ml_matmul_batch(input, (UIntPtr)input.Length, Ptr, output, (UIntPtr)output.Length, (UIntPtr)batchSize);

        public void Dispose() { if (Ptr != IntPtr.Zero) { Native.am_ml_weight_free(Ptr); Ptr = IntPtr.Zero; } }
    }

    /// Bit-parallel kernel handle (SIMD-ready)
    public sealed class TernaryKernelHandle : IDisposable
    {
        internal IntPtr Ptr;
        internal bool Consumed;
        internal TernaryKernelHandle(IntPtr ptr) { Ptr = ptr; }

        public static TernaryKernelHandle FromTernary(sbyte[] values, int outFeatures, int inFeatures)
            => new TernaryKernelHandle(Native.am_ml_kernel_from_ternary(values, (UIntPtr)values.Length, (UIntPtr)outFeatures, (UIntPtr)inFeatures));

        public static TernaryKernelHandle FromTernaryScaled(sbyte[] values, int outFeatures, int inFeatures, float scale)
            => new TernaryKernelHandle(Native.am_ml_kernel_from_ternary_scaled(values, (UIntPtr)values.Length, (UIntPtr)outFeatures, (UIntPtr)inFeatures, scale));

        public static TernaryKernelHandle FromWeight(TernaryWeightHandle w)
            => new TernaryKernelHandle(Native.am_ml_kernel_from_weight(w.Ptr));

        public int OutFeatures => (int)Native.am_ml_kernel_out_features(Ptr);
        public int InFeatures => (int)Native.am_ml_kernel_in_features(Ptr);
        public int MemoryBytes => (int)Native.am_ml_kernel_memory_bytes(Ptr);
        public float CompressionRatio => Native.am_ml_kernel_compression_ratio(Ptr);

        public void MatvecKernel(float[] input, float[] output)
            => Native.am_ml_matvec_kernel(input, (UIntPtr)input.Length, Ptr, output, (UIntPtr)output.Length);

        public void MatvecSimd(float[] input, float[] output)
            => Native.am_ml_matvec_simd(input, (UIntPtr)input.Length, Ptr, output, (UIntPtr)output.Length);

        /// Mark as consumed (ownership transferred to BitLinearHandle)
        internal void MarkConsumed() { Consumed = true; Ptr = IntPtr.Zero; }

        public void Dispose() { if (!Consumed && Ptr != IntPtr.Zero) { Native.am_ml_kernel_free(Ptr); Ptr = IntPtr.Zero; } }
    }

    /// BitLinear neural layer handle
    public sealed class BitLinearHandle : IDisposable
    {
        internal IntPtr Ptr;
        internal BitLinearHandle(IntPtr ptr) { Ptr = ptr; }

        /// Create BitLinear layer (consumes kernel — do not use kernel after this call)
        public static BitLinearHandle Create(TernaryKernelHandle kernel, float[] bias, bool preNorm)
        {
            var handle = new BitLinearHandle(Native.am_ml_bitlinear_new(kernel.Ptr, bias, bias != null ? (UIntPtr)bias.Length : UIntPtr.Zero, preNorm ? 1 : 0));
            kernel.MarkConsumed();
            return handle;
        }

        public void Forward(float[] input, float[] output)
            => Native.am_ml_bitlinear_forward(Ptr, input, (UIntPtr)input.Length, output, (UIntPtr)output.Length);

        public int MemoryBytes => (int)Native.am_ml_bitlinear_memory_bytes(Ptr);
        public float CompressionRatio => Native.am_ml_bitlinear_compression_ratio(Ptr);

        public void Dispose() { if (Ptr != IntPtr.Zero) { Native.am_ml_bitlinear_free(Ptr); Ptr = IntPtr.Zero; } }
    }

    /// Quantization result handle
    public sealed class QuantizedHandle : IDisposable
    {
        internal IntPtr Ptr;
        private float[] _original;
        private int _len;

        public static QuantizedHandle Quantize(float[] weights, int outFeatures, int inFeatures)
        {
            var h = new QuantizedHandle();
            h.Ptr = Native.am_ml_quantize(weights, (UIntPtr)weights.Length, (UIntPtr)outFeatures, (UIntPtr)inFeatures);
            h._original = weights;
            h._len = weights.Length;
            return h;
        }

        public float[] Dequantize()
        {
            var buf = new float[_len];
            Native.am_ml_dequantize(Ptr, buf, (UIntPtr)buf.Length);
            return buf;
        }

        public float MAE => Native.am_ml_quantization_error_mae(_original, (UIntPtr)_len, Ptr);
        public float SNR => Native.am_ml_quantization_error_snr(_original, (UIntPtr)_len, Ptr);

        public void Dispose() { if (Ptr != IntPtr.Zero) { Native.am_ml_weight_free(Ptr); Ptr = IntPtr.Zero; } }
    }

    // ========================================================================
    // Static Tensor Ops
    // ========================================================================

    /// Element-wise tensor operations (DPS, zero allocation)
    public static class TensorOps
    {
        public static void Add(float[] a, float[] b, float[] output) => Native.am_ml_tensor_add(a, b, output, (UIntPtr)a.Length);
        public static void Sub(float[] a, float[] b, float[] output) => Native.am_ml_tensor_sub(a, b, output, (UIntPtr)a.Length);
        public static void Scale(float[] a, float s, float[] output) => Native.am_ml_tensor_scale(a, s, output, (UIntPtr)a.Length);
        public static void Copy(float[] a, float[] output) => Native.am_ml_tensor_copy(a, output, (UIntPtr)a.Length);
        public static float Sum(float[] a) => Native.am_ml_tensor_sum(a, (UIntPtr)a.Length);
        public static float Mean(float[] a) => Native.am_ml_tensor_mean(a, (UIntPtr)a.Length);
        public static float Min(float[] a) => Native.am_ml_tensor_min(a, (UIntPtr)a.Length);
        public static float Max(float[] a) => Native.am_ml_tensor_max(a, (UIntPtr)a.Length);
        public static void ReLU(float[] a, float[] output) => Native.am_ml_tensor_relu(a, output, (UIntPtr)a.Length);
        public static void ReLUInplace(float[] a) => Native.am_ml_tensor_relu_inplace(a, (UIntPtr)a.Length);
        public static void Softmax(float[] a, float[] output) => Native.am_ml_tensor_softmax(a, output, (UIntPtr)a.Length);
        public static void RmsNorm(float[] a, float epsilon, float[] output) => Native.am_ml_tensor_rms_norm(a, epsilon, output, (UIntPtr)a.Length);
        public static void LayerNorm(float[] a, float epsilon, float[] output) => Native.am_ml_tensor_layer_norm(a, epsilon, output, (UIntPtr)a.Length);
    }

    /// Version info
    public static class AliceMl
    {
        public static string Version => Marshal.PtrToStringAnsi(Native.am_ml_version());
    }
}
