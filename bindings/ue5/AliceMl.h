// ALICE-ML UE5 C++ Bindings
// 65 extern "C" + 9 RAII handles
//
// Author: Moroya Sakamoto

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

// ============================================================================
// C-ABI (65 functions)
// ============================================================================

extern "C"
{
    // ---- Arena (7) ----
    void* am_ml_arena_new(size_t capacity);
    void  am_ml_arena_free(void* arena);
    void  am_ml_arena_reset(void* arena);
    size_t am_ml_arena_used(const void* arena);
    size_t am_ml_arena_capacity(const void* arena);
    size_t am_ml_arena_remaining(const void* arena);
    float* am_ml_arena_alloc_f32(void* arena, size_t count);

    // ---- TernaryWeight (8) ----
    void* am_ml_weight_from_ternary(const int8_t* values, size_t len, size_t out_features, size_t in_features);
    void  am_ml_weight_free(void* w);
    size_t am_ml_weight_out_features(const void* w);
    size_t am_ml_weight_in_features(const void* w);
    float  am_ml_weight_scale(const void* w);
    size_t am_ml_weight_memory_bytes(const void* w);
    float  am_ml_weight_compression_ratio(const void* w);
    int8_t am_ml_weight_get(const void* w, size_t row, size_t col);

    // ---- TernaryWeightKernel (9) ----
    void* am_ml_kernel_from_ternary(const int8_t* values, size_t len, size_t out_features, size_t in_features);
    void* am_ml_kernel_from_ternary_scaled(const int8_t* values, size_t len, size_t out_features, size_t in_features, float scale);
    void* am_ml_kernel_from_weight(const void* w);
    void  am_ml_kernel_free(void* k);
    size_t am_ml_kernel_out_features(const void* k);
    size_t am_ml_kernel_in_features(const void* k);
    size_t am_ml_kernel_memory_bytes(const void* k);
    float  am_ml_kernel_compression_ratio(const void* k);
    size_t am_ml_kernel_words_per_row(const void* k);

    // ---- Matvec DPS (4) ----
    void am_ml_matvec(const float* input, size_t in_len, const void* weights, float* output, size_t out_len);
    void am_ml_matvec_kernel(const float* input, size_t in_len, const void* kernel, float* output, size_t out_len);
    void am_ml_matmul_batch(const float* input, size_t in_len, const void* weights, float* output, size_t out_len, size_t batch_size);
    void am_ml_matvec_simd(const float* input, size_t in_len, const void* kernel, float* output, size_t out_len);

    // ---- Tensor DPS ops (13) ----
    void  am_ml_tensor_add(const float* a, const float* b, float* out, size_t len);
    void  am_ml_tensor_sub(const float* a, const float* b, float* out, size_t len);
    void  am_ml_tensor_scale(const float* a, float s, float* out, size_t len);
    void  am_ml_tensor_copy(const float* a, float* out, size_t len);
    float am_ml_tensor_sum(const float* a, size_t len);
    float am_ml_tensor_mean(const float* a, size_t len);
    float am_ml_tensor_min(const float* a, size_t len);
    float am_ml_tensor_max(const float* a, size_t len);
    void  am_ml_tensor_relu(const float* a, float* out, size_t len);
    void  am_ml_tensor_relu_inplace(float* a, size_t len);
    void  am_ml_tensor_softmax(const float* a, float* out, size_t len);
    void  am_ml_tensor_rms_norm(const float* a, float epsilon, float* out, size_t len);
    void  am_ml_tensor_layer_norm(const float* a, float epsilon, float* out, size_t len);

    // ---- BitLinear (5) ----
    void*  am_ml_bitlinear_new(void* kernel, const float* bias, size_t bias_len, int pre_norm);
    void   am_ml_bitlinear_free(void* layer);
    void   am_ml_bitlinear_forward(const void* layer, const float* input, size_t in_len, float* output, size_t out_len);
    size_t am_ml_bitlinear_memory_bytes(const void* layer);
    float  am_ml_bitlinear_compression_ratio(const void* layer);

    // ---- Quantize (4) ----
    void*  am_ml_quantize(const float* weights, size_t len, size_t out_features, size_t in_features);
    size_t am_ml_dequantize(const void* w, float* out, size_t max_len);
    float  am_ml_quantization_error_mae(const float* original, size_t len, const void* quantized);
    float  am_ml_quantization_error_snr(const float* original, size_t len, const void* quantized);

    // ---- MicroModel (8) ----
    void*  am_ml_micro_model_build_random(size_t in_features, size_t out_features, const size_t* hidden_dims, size_t hidden_count, size_t budget_bytes, uint64_t seed);
    void   am_ml_micro_model_free(void* model);
    void   am_ml_micro_model_forward(const void* model, const float* input, size_t in_len, float* output, size_t out_len);
    size_t am_ml_micro_model_predict_tokens(const void* model, const float* input, size_t in_len, float* logits, size_t logits_len, size_t steps);
    size_t am_ml_micro_model_memory_bytes(const void* model);
    int    am_ml_micro_model_fits_in_budget(const void* model);
    size_t am_ml_micro_model_param_count(const void* model);
    size_t am_ml_micro_model_depth(const void* model);

    // ---- CacheResidentDecoder (6) ----
    void*  am_ml_cache_decoder_new(void* draft, void* verify_kernel, size_t max_draft_tokens);
    void   am_ml_cache_decoder_free(void* decoder);
    size_t am_ml_cache_decoder_step(const void* decoder, const float* input, size_t in_len, float* draft_buf, size_t draft_len, float* verify_buf, size_t verify_len);
    int    am_ml_cache_decoder_fits_in_cache(const void* decoder);
    size_t am_ml_cache_decoder_draft_memory(const void* decoder);
    size_t am_ml_cache_decoder_verify_memory(const void* decoder);

    // ---- Version (1) ----
    const char* am_ml_version();
}

// ============================================================================
// RAII Handles (C++)
// ============================================================================

namespace AliceMl
{

/// Arena bump allocator
struct ArenaDeleter { void operator()(void* p) const { am_ml_arena_free(p); } };
using ArenaPtr = std::unique_ptr<void, ArenaDeleter>;

inline ArenaPtr MakeArena(size_t capacity) { return ArenaPtr(am_ml_arena_new(capacity)); }

/// Packed ternary weight
struct WeightDeleter { void operator()(void* p) const { am_ml_weight_free(p); } };
using WeightPtr = std::unique_ptr<void, WeightDeleter>;

inline WeightPtr MakeWeight(const int8_t* values, size_t len, size_t out_f, size_t in_f)
{
    return WeightPtr(am_ml_weight_from_ternary(values, len, out_f, in_f));
}

/// Bit-parallel kernel (SIMD-ready)
struct KernelDeleter { void operator()(void* p) const { am_ml_kernel_free(p); } };
using KernelPtr = std::unique_ptr<void, KernelDeleter>;

inline KernelPtr MakeKernel(const int8_t* values, size_t len, size_t out_f, size_t in_f)
{
    return KernelPtr(am_ml_kernel_from_ternary(values, len, out_f, in_f));
}

inline KernelPtr MakeKernelScaled(const int8_t* values, size_t len, size_t out_f, size_t in_f, float scale)
{
    return KernelPtr(am_ml_kernel_from_ternary_scaled(values, len, out_f, in_f, scale));
}

inline KernelPtr MakeKernelFromWeight(const WeightPtr& w)
{
    return KernelPtr(am_ml_kernel_from_weight(w.get()));
}

/// BitLinear neural layer
struct BitLinearDeleter { void operator()(void* p) const { am_ml_bitlinear_free(p); } };
using BitLinearPtr = std::unique_ptr<void, BitLinearDeleter>;

/// Create BitLinear layer (consumes kernel — do not use kernel after this call)
inline BitLinearPtr MakeBitLinear(KernelPtr& kernel, const float* bias, size_t bias_len, bool pre_norm)
{
    auto layer = BitLinearPtr(am_ml_bitlinear_new(kernel.release(), bias, bias_len, pre_norm ? 1 : 0));
    return layer;
}

/// Forward pass
inline void Forward(const BitLinearPtr& layer, const float* input, size_t in_len, float* output, size_t out_len)
{
    am_ml_bitlinear_forward(layer.get(), input, in_len, output, out_len);
}

/// Matvec with packed weights
inline void Matvec(const WeightPtr& w, const float* input, size_t in_len, float* output, size_t out_len)
{
    am_ml_matvec(input, in_len, w.get(), output, out_len);
}

/// Matvec with SIMD kernel
inline void MatvecSimd(const KernelPtr& k, const float* input, size_t in_len, float* output, size_t out_len)
{
    am_ml_matvec_simd(input, in_len, k.get(), output, out_len);
}

/// Batch matmul
inline void MatmulBatch(const WeightPtr& w, const float* input, size_t in_len, float* output, size_t out_len, size_t batch_size)
{
    am_ml_matmul_batch(input, in_len, w.get(), output, out_len, batch_size);
}

/// Quantize FP32 weights to ternary
inline WeightPtr Quantize(const float* weights, size_t len, size_t out_f, size_t in_f)
{
    return WeightPtr(am_ml_quantize(weights, len, out_f, in_f));
}

/// Dequantize ternary to FP32
inline std::vector<float> Dequantize(const WeightPtr& w, size_t max_len)
{
    std::vector<float> buf(max_len);
    size_t n = am_ml_dequantize(w.get(), buf.data(), max_len);
    buf.resize(n);
    return buf;
}

/// L2 cache-resident micro model
struct MicroModelDeleter { void operator()(void* p) const { am_ml_micro_model_free(p); } };
using MicroModelPtr = std::unique_ptr<void, MicroModelDeleter>;

inline MicroModelPtr MakeMicroModel(size_t in_f, size_t out_f, const size_t* hidden, size_t hidden_count, size_t budget, uint64_t seed)
{
    return MicroModelPtr(am_ml_micro_model_build_random(in_f, out_f, hidden, hidden_count, budget, seed));
}

/// L2 cache-resident speculative decoder
struct CacheDecoderDeleter { void operator()(void* p) const { am_ml_cache_decoder_free(p); } };
using CacheDecoderPtr = std::unique_ptr<void, CacheDecoderDeleter>;

/// Create cache-resident decoder (consumes draft model and verify kernel)
inline CacheDecoderPtr MakeCacheDecoder(MicroModelPtr& draft, KernelPtr& verify_kernel, size_t max_draft_tokens)
{
    return CacheDecoderPtr(am_ml_cache_decoder_new(draft.release(), verify_kernel.release(), max_draft_tokens));
}

/// Get version string
inline std::string Version()
{
    return std::string(am_ml_version());
}

} // namespace AliceMl
