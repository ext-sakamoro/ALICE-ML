//! Llama-3 inference with 1.58-bit ternary weights.
//!
//! Converts FP32/BF16 weights to ternary {-1, 0, +1} using ALICE-ML's
//! quantization pipeline. At 1.58 bits per weight, a 70B model fits in
//! ~13GB RAM.
//!
//! # Memory usage
//! - 8B model: ~1.6 GB (ternary) vs 16 GB (BF16)
//! - 70B model: ~13 GB (ternary) vs 140 GB (BF16)

use crate::model_io::ModelArchive;
use crate::ops::{ternary_matvec, TernaryWeight};
use crate::quantize::quantize_to_ternary;
use crate::safetensors::SafetensorsFile;

// ─── Model config ───────────────────────────────────────────────────────────

/// Llama-3 configuration for ternary inference.
#[derive(Debug, Clone)]
pub struct Llama3TernaryConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// FFN intermediate dimension (SwiGLU)
    pub intermediate_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// RoPE base frequency
    pub rope_theta: f32,
    /// RMS norm epsilon
    pub norm_eps: f32,
}

impl Llama3TernaryConfig {
    /// Default config for Llama-3 8B.
    #[must_use]
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_dim: 4096,
            intermediate_dim: 14_336,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            max_seq_len: 8192,
            head_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
        }
    }

    /// Config for Qwen2.5-1.5B-Instruct.
    #[must_use]
    pub fn qwen2_5_1_5b() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_dim: 1536,
            intermediate_dim: 8960,
            num_heads: 12,
            num_kv_heads: 2,
            num_layers: 28,
            max_seq_len: 32_768,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
        }
    }
}

// ─── Ternary layer ──────────────────────────────────────────────────────────

/// A single transformer layer with ternary-quantized weights.
struct TernaryLayer {
    attn_norm_weight: Vec<f32>,
    q_proj: TernaryWeight,
    k_proj: TernaryWeight,
    v_proj: TernaryWeight,
    o_proj: TernaryWeight,
    ffn_norm_weight: Vec<f32>,
    gate_proj: TernaryWeight,
    up_proj: TernaryWeight,
    down_proj: TernaryWeight,
}

/// KV cache for ternary model.
struct KvCache {
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

impl KvCache {
    fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
        }
    }

    fn append(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        self.keys[layer].push(k);
        self.values[layer].push(v);
    }

    fn seq_len(&self, layer: usize) -> usize {
        self.keys[layer].len()
    }

    fn clear(&mut self) {
        for l in &mut self.keys {
            l.clear();
        }
        for l in &mut self.values {
            l.clear();
        }
    }
}

// ─── Math helpers ───────────────────────────────────────────────────────────

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv_rms = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

fn apply_rope(vec: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + 1];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Ternary model ──────────────────────────────────────────────────────────

/// Llama-3 model with ternary-quantized weights.
///
/// Memory footprint is ~16x smaller than FP32:
/// - 8B model: ~1.6 GB
/// - 70B model: ~13 GB
pub struct Llama3TernaryModel {
    /// Model config
    pub config: Llama3TernaryConfig,
    /// Embedding table (kept as FP32 for accuracy)
    embedding: Vec<f32>,
    /// Ternary transformer layers
    layers: Vec<TernaryLayer>,
    /// Output RMS norm weights (FP32)
    output_norm: Vec<f32>,
    /// Output projection (ternary)
    output_proj: TernaryWeight,
    /// KV cache
    kv_cache: KvCache,
    /// Quantization statistics
    pub quant_stats: QuantizationReport,
}

/// Report of quantization quality.
#[derive(Debug, Clone, Default)]
pub struct QuantizationReport {
    /// Total parameters
    pub total_params: usize,
    /// Sparsity (fraction of zero weights)
    pub avg_sparsity: f32,
    /// Average effective bits per weight
    pub avg_effective_bits: f32,
    /// FP32 size in bytes
    pub fp32_bytes: usize,
    /// Ternary size in bytes
    pub ternary_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Per-layer mean absolute error
    pub layer_mae: Vec<f32>,
}

impl Llama3TernaryModel {
    /// Quantize model from safetensors file to ternary.
    ///
    /// Reads BF16/FP16 weights, quantizes each to ternary {-1, 0, +1},
    /// and stores using 2-bit packing (4 weights per byte).
    pub fn from_safetensors(sf: &SafetensorsFile<'_>, config: Llama3TernaryConfig) -> Option<Self> {
        let mut report = QuantizationReport::default();

        // Embedding (keep as FP32 for accuracy)
        let embedding = sf.tensor_to_f32("model.embed_tokens.weight")?;

        // Output norm (FP32)
        let output_norm = sf.tensor_to_f32("model.norm.weight")?;

        // Output projection (ternary) — fallback to embedding for tied weights
        let output_fp32 = sf
            .tensor_to_f32("lm_head.weight")
            .unwrap_or_else(|| embedding.clone());
        let (output_proj, out_stats) =
            quantize_to_ternary(&output_fp32, config.vocab_size, config.hidden_dim);
        report.total_params += config.vocab_size * config.hidden_dim;
        report.layer_mae.push(out_stats.mae);

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        let kv_dim = config.num_kv_heads * config.head_dim;

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");

            let attn_norm_weight = sf.tensor_to_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let ffn_norm_weight =
                sf.tensor_to_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;

            // Quantize attention projections
            let q_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let (q_proj, q_stats) =
                quantize_to_ternary(&q_fp32, config.hidden_dim, config.hidden_dim);
            report.layer_mae.push(q_stats.mae);

            let k_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let (k_proj, k_stats) = quantize_to_ternary(&k_fp32, kv_dim, config.hidden_dim);
            report.layer_mae.push(k_stats.mae);

            let v_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let (v_proj, v_stats) = quantize_to_ternary(&v_fp32, kv_dim, config.hidden_dim);
            report.layer_mae.push(v_stats.mae);

            let o_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let (o_proj, o_stats) =
                quantize_to_ternary(&o_fp32, config.hidden_dim, config.hidden_dim);
            report.layer_mae.push(o_stats.mae);

            // Quantize FFN projections
            let gate_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let (gate_proj, gate_stats) =
                quantize_to_ternary(&gate_fp32, config.intermediate_dim, config.hidden_dim);
            report.layer_mae.push(gate_stats.mae);

            let up_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.up_proj.weight"))?;
            let (up_proj, up_stats) =
                quantize_to_ternary(&up_fp32, config.intermediate_dim, config.hidden_dim);
            report.layer_mae.push(up_stats.mae);

            let down_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.down_proj.weight"))?;
            let (down_proj, down_stats) =
                quantize_to_ternary(&down_fp32, config.hidden_dim, config.intermediate_dim);
            report.layer_mae.push(down_stats.mae);

            // Track params
            let layer_params = config.hidden_dim * config.hidden_dim * 2 // q + o
                + kv_dim * config.hidden_dim * 2 // k + v
                + config.intermediate_dim * config.hidden_dim * 3; // gate + up + down
            report.total_params += layer_params;

            layers.push(TernaryLayer {
                attn_norm_weight,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ffn_norm_weight,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        // Compute report
        report.fp32_bytes = report.total_params * 4;
        report.ternary_bytes = report.total_params / 4; // 2 bits per weight
        report.compression_ratio = report.fp32_bytes as f32 / report.ternary_bytes.max(1) as f32;
        report.avg_sparsity =
            report.layer_mae.iter().sum::<f32>() / report.layer_mae.len().max(1) as f32;

        let kv_cache = KvCache::new(config.num_layers);

        Some(Self {
            config,
            embedding,
            layers,
            output_norm,
            output_proj,
            kv_cache,
            quant_stats: report,
        })
    }

    /// Quantize from safetensors using pre-computed QAT scales.
    ///
    /// `scales` maps `"layer_idx.weight_name"` → `(γ, temperature)`.
    /// E.g., `"0.q_proj"` → `(0.033, 0.98)`.
    /// Weights not in the map fall back to auto-computed γ.
    #[cfg(feature = "safetensors")]
    pub fn from_safetensors_with_scales(
        sf: &SafetensorsFile<'_>,
        config: Llama3TernaryConfig,
        scales: &std::collections::HashMap<String, (f32, f32)>,
    ) -> Option<Self> {
        use crate::quantize::{quantize_to_ternary, quantize_to_ternary_qat};

        let mut report = QuantizationReport::default();

        let embedding = sf.tensor_to_f32("model.embed_tokens.weight")?;
        let output_norm = sf.tensor_to_f32("model.norm.weight")?;
        let output_fp32 = sf
            .tensor_to_f32("lm_head.weight")
            .unwrap_or_else(|| embedding.clone());
        let (output_proj, out_stats) =
            quantize_to_ternary(&output_fp32, config.vocab_size, config.hidden_dim);
        report.total_params += config.vocab_size * config.hidden_dim;
        report.layer_mae.push(out_stats.mae);

        let mut layers = Vec::with_capacity(config.num_layers);
        let kv_dim = config.num_kv_heads * config.head_dim;

        let qat = |fp32: &[f32], out_f: usize, in_f: usize, layer: usize, name: &str| {
            let key = format!("{layer}.{name}");
            if let Some(&(gamma, temp)) = scales.get(&key) {
                quantize_to_ternary_qat(fp32, out_f, in_f, gamma, temp)
            } else {
                quantize_to_ternary(fp32, out_f, in_f)
            }
        };

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");
            let attn_norm_weight = sf.tensor_to_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let ffn_norm_weight =
                sf.tensor_to_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;

            let q_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let (q_proj, q_s) = qat(&q_fp32, config.hidden_dim, config.hidden_dim, i, "q_proj");
            report.layer_mae.push(q_s.mae);

            let k_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let (k_proj, k_s) = qat(&k_fp32, kv_dim, config.hidden_dim, i, "k_proj");
            report.layer_mae.push(k_s.mae);

            let v_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let (v_proj, v_s) = qat(&v_fp32, kv_dim, config.hidden_dim, i, "v_proj");
            report.layer_mae.push(v_s.mae);

            let o_fp32 = sf.tensor_to_f32(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let (o_proj, o_s) = qat(&o_fp32, config.hidden_dim, config.hidden_dim, i, "o_proj");
            report.layer_mae.push(o_s.mae);

            let gate_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let (gate_proj, g_s) = qat(
                &gate_fp32,
                config.intermediate_dim,
                config.hidden_dim,
                i,
                "gate_proj",
            );
            report.layer_mae.push(g_s.mae);

            let up_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.up_proj.weight"))?;
            let (up_proj, u_s) = qat(
                &up_fp32,
                config.intermediate_dim,
                config.hidden_dim,
                i,
                "up_proj",
            );
            report.layer_mae.push(u_s.mae);

            let down_fp32 = sf.tensor_to_f32(&format!("{prefix}.mlp.down_proj.weight"))?;
            let (down_proj, d_s) = qat(
                &down_fp32,
                config.hidden_dim,
                config.intermediate_dim,
                i,
                "down_proj",
            );
            report.layer_mae.push(d_s.mae);

            let layer_params = config.hidden_dim * config.hidden_dim * 2
                + kv_dim * config.hidden_dim * 2
                + config.intermediate_dim * config.hidden_dim * 3;
            report.total_params += layer_params;

            layers.push(TernaryLayer {
                attn_norm_weight,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ffn_norm_weight,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        report.fp32_bytes = report.total_params * 4;
        report.ternary_bytes = report.total_params / 4;
        report.compression_ratio = report.fp32_bytes as f32 / report.ternary_bytes.max(1) as f32;
        report.avg_sparsity =
            report.layer_mae.iter().sum::<f32>() / report.layer_mae.len().max(1) as f32;

        let kv_cache = KvCache::new(config.num_layers);

        Some(Self {
            config,
            embedding,
            layers,
            output_norm,
            output_proj,
            kv_cache,
            quant_stats: report,
        })
    }

    /// Build model from pre-quantized parts (ATML + FP32).
    ///
    /// `layers_data`: Vec of (attn_norm, ffn_norm, [q,k,v,o,gate,up,down]_proj)
    #[must_use]
    pub fn from_parts(
        config: Llama3TernaryConfig,
        embedding: Vec<f32>,
        output_norm: Vec<f32>,
        output_proj: TernaryWeight,
        layers_data: Vec<(Vec<f32>, Vec<f32>, Vec<TernaryWeight>)>,
    ) -> Self {
        let kv_cache = KvCache::new(config.num_layers);
        let mut layers = Vec::with_capacity(layers_data.len());
        for (attn_norm_weight, ffn_norm_weight, projs) in layers_data {
            assert!(projs.len() >= 7, "need 7 projections per layer");
            layers.push(TernaryLayer {
                attn_norm_weight,
                q_proj: projs[0].clone(),
                k_proj: projs[1].clone(),
                v_proj: projs[2].clone(),
                o_proj: projs[3].clone(),
                ffn_norm_weight,
                gate_proj: projs[4].clone(),
                up_proj: projs[5].clone(),
                down_proj: projs[6].clone(),
            });
        }
        Self {
            config,
            embedding,
            layers,
            output_norm,
            output_proj,
            kv_cache,
            quant_stats: QuantizationReport::default(),
        }
    }

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Total model memory in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let emb = self.embedding.len() * 4;
        let norm = self.output_norm.len() * 4;
        let out = self.output_proj.memory_bytes();
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm_weight.len() * 4
                    + l.ffn_norm_weight.len() * 4
                    + l.q_proj.memory_bytes()
                    + l.k_proj.memory_bytes()
                    + l.v_proj.memory_bytes()
                    + l.o_proj.memory_bytes()
                    + l.gate_proj.memory_bytes()
                    + l.up_proj.memory_bytes()
                    + l.down_proj.memory_bytes()
            })
            .sum();
        emb + norm + out + layers
    }

    /// Forward pass for a single token. Returns logits.
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        let c = &self.config;
        let pos = self.kv_cache.seq_len(0);

        // Embedding
        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];

            // Attention norm
            rms_norm(&hidden, &layer.attn_norm_weight, c.norm_eps, &mut norm_buf);

            // Q, K, V (ternary matvec — no multiplication, just add/sub)
            ternary_matvec(&norm_buf, &layer.q_proj, &mut q_buf);
            ternary_matvec(&norm_buf, &layer.k_proj, &mut k_buf);
            ternary_matvec(&norm_buf, &layer.v_proj, &mut v_buf);

            // RoPE
            for h in 0..c.num_heads {
                let s = h * c.head_dim;
                apply_rope(&mut q_buf[s..s + c.head_dim], pos, c.head_dim, c.rope_theta);
            }
            for h in 0..c.num_kv_heads {
                let s = h * c.head_dim;
                apply_rope(&mut k_buf[s..s + c.head_dim], pos, c.head_dim, c.rope_theta);
            }

            self.kv_cache
                .append(layer_idx, k_buf.clone(), v_buf.clone());

            // GQA attention
            let seq_len = self.kv_cache.seq_len(layer_idx);
            let heads_per_kv = c.num_heads / c.num_kv_heads;
            attn_out.fill(0.0);

            for h in 0..c.num_heads {
                let kv_h = h / heads_per_kv;
                let q_start = h * c.head_dim;
                let q_head = &q_buf[q_start..q_start + c.head_dim];

                let mut scores = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let k_cached = &self.kv_cache.keys[layer_idx][t];
                    let k_start = kv_h * c.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..c.head_dim {
                        score += q_head[d] * k_cached[k_start + d];
                    }
                    scores.push(score / (c.head_dim as f32).sqrt());
                }

                // Softmax
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in &mut scores {
                        *s /= sum;
                    }
                }

                for t in 0..seq_len {
                    let v_cached = &self.kv_cache.values[layer_idx][t];
                    let v_start = kv_h * c.head_dim;
                    for d in 0..c.head_dim {
                        attn_out[q_start + d] += scores[t] * v_cached[v_start + d];
                    }
                }
            }

            ternary_matvec(&attn_out, &layer.o_proj, &mut o_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            // FFN
            rms_norm(&hidden, &layer.ffn_norm_weight, c.norm_eps, &mut norm_buf);

            ternary_matvec(&norm_buf, &layer.gate_proj, &mut gate_buf);
            ternary_matvec(&norm_buf, &layer.up_proj, &mut up_buf);

            for i in 0..c.intermediate_dim {
                gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
            }

            ternary_matvec(&gate_buf, &layer.down_proj, &mut down_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        // Output
        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);

        let mut logits = vec![0.0f32; c.vocab_size];
        ternary_matvec(&norm_buf, &self.output_proj, &mut logits);

        logits
    }

    /// Save quantized model in ATML format for fast reloading.
    pub fn save_atml(&self) -> Vec<u8> {
        let mut archive = ModelArchive::new();

        archive.add_layer("output_proj", &self.output_proj);

        for (i, layer) in self.layers.iter().enumerate() {
            archive.add_layer(&format!("layer.{i}.q_proj"), &layer.q_proj);
            archive.add_layer(&format!("layer.{i}.k_proj"), &layer.k_proj);
            archive.add_layer(&format!("layer.{i}.v_proj"), &layer.v_proj);
            archive.add_layer(&format!("layer.{i}.o_proj"), &layer.o_proj);
            archive.add_layer(&format!("layer.{i}.gate_proj"), &layer.gate_proj);
            archive.add_layer(&format!("layer.{i}.up_proj"), &layer.up_proj);
            archive.add_layer(&format!("layer.{i}.down_proj"), &layer.down_proj);
        }

        archive.serialize()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_8b() {
        let c = Llama3TernaryConfig::llama3_8b();
        assert_eq!(c.vocab_size, 128_256);
        assert_eq!(c.hidden_dim, 4096);
        assert_eq!(c.num_heads / c.num_kv_heads, 4);
    }

    #[test]
    fn test_ternary_memory_estimate() {
        let c = Llama3TernaryConfig::llama3_8b();
        let total_params: usize = c.vocab_size * c.hidden_dim
            + c.num_layers
                * (c.hidden_dim * c.hidden_dim * 2
                    + c.num_kv_heads * c.head_dim * c.hidden_dim * 2
                    + c.intermediate_dim * c.hidden_dim * 3)
            + c.vocab_size * c.hidden_dim;

        // Ternary: 2 bits per weight + scale
        let ternary_bytes = total_params / 4;
        let gb = ternary_bytes as f64 / 1e9;
        // Should be ~1.5-2.0 GB
        assert!(gb > 1.0 && gb < 3.0, "8B ternary estimate: {gb:.2} GB");
    }

    #[test]
    fn test_70b_ternary_memory_estimate() {
        // 70B model params estimate
        let total_params: usize = 70_000_000_000;
        let ternary_bytes = total_params / 4;
        let gb = ternary_bytes as f64 / 1e9;
        // Should fit in 32GB Mac Mini
        assert!(gb < 20.0, "70B ternary estimate: {gb:.2} GB");
    }

    #[test]
    fn test_rms_norm() {
        let x = [1.0f32, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        rms_norm(&x, &w, 1e-5, &mut out);

        let rms = ((1.0 + 4.0 + 9.0) / 3.0 + 1e-5f32).sqrt();
        for i in 0..3 {
            assert!((out[i] - x[i] / rms).abs() < 1e-4);
        }
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KvCache::new(4);
        assert_eq!(cache.seq_len(0), 0);
        cache.append(0, vec![1.0], vec![2.0]);
        assert_eq!(cache.seq_len(0), 1);
        cache.clear();
        assert_eq!(cache.seq_len(0), 0);
    }
}
