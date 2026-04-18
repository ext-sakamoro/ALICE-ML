//! ALICE QAT ECU テスト — .atml から直接ロードして推論
//!
//! 正しいフロー: QATで学習済みの三値重みを .atml から読み込み、
//! ALICE-ML の ternary_matvec で推論。量子化ステップは不要。
//!
//! ```bash
//! cargo run --example qwen_qat_test --features safetensors --release -- \
//!   --atml ~/Project-ALICE/models/qwen2.5-1.5b-qat-merged/qwen2.5-1.5b-qat.atml \
//!   --fp32 ~/Project-ALICE/models/qwen2.5-1.5b-qat-merged/qwen2.5-1.5b-qat-fp32.bin \
//!   --tokenizer ~/Project-ALICE/models/qwen2.5-1.5b-qat-merged/tokenizer.json
//! ```

use alice_ml::llama3_ternary::{Llama3TernaryConfig, Llama3TernaryModel};
use alice_ml::model_io::ModelArchive;
use alice_ml::ops::TernaryWeight;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    let get_arg = |flag: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .unwrap_or_else(|| panic!("Missing {flag}"))
            .clone()
    };

    let atml_path = get_arg("--atml");
    let fp32_path = get_arg("--fp32");
    let tokenizer_path = get_arg("--tokenizer");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  ALICE QAT ECU — Load .atml + Real Inference            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let start = Instant::now();

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

    // Load FP32 data (embedding, norms, biases)
    println!("Loading FP32 data: {fp32_path}");
    let fp32_raw = fs::read(&fp32_path).expect("Failed to read FP32 file");
    let fp32_data = parse_fp32_bin(&fp32_raw);
    println!(
        "  {} entries ({:.1} MB)",
        fp32_data.len(),
        fp32_raw.len() as f64 / 1e6
    );

    // Load ATML
    println!("Loading ATML: {atml_path}");
    let atml_raw = fs::read(&atml_path).expect("Failed to read ATML file");
    let archive = ModelArchive::deserialize(&atml_raw).expect("Failed to parse ATML");
    println!(
        "  {} layers ({:.1} MB)",
        archive.layer_count(),
        atml_raw.len() as f64 / 1e6
    );

    // Build model from ATML + FP32
    println!("Building model...");
    let config = Llama3TernaryConfig::qwen2_5_1_5b();
    let mut model = build_model_from_atml(&archive, &fp32_data, &config);
    println!(
        "  Model memory: {:.1} MB",
        model.memory_bytes() as f64 / 1e6
    );
    println!("  Load time: {:.1}s", start.elapsed().as_secs_f32());

    // ECU prompts
    let prompts = vec![
        (
            "You are ALICE AGI. Use tools when needed.",
            "128 × 256 はいくつ？",
            "math",
        ),
        (
            "You are ALICE AGI. Use vectordb_search for facts.",
            "2型糖尿病の最新治療法は？",
            "medical",
        ),
        (
            "You are ALICE AGI. Think step by step.",
            "なぜ金利が上がるとハイテク株は下がる？",
            "finance",
        ),
        ("You are ALICE AGI.", "Hello, what is Rust?", "code"),
    ];

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  ECU Inference (from .atml, no re-quantization)          ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    for (system, user, category) in &prompts {
        println!("\n--- [{category}] Q: {user} ---");

        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        );

        let encoding = tokenizer.encode(prompt.as_str(), false).expect("encode");
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        model.clear_cache();

        // Prefill
        let pf_start = Instant::now();
        let mut logits = Vec::new();
        for &tid in &input_ids {
            logits = model.forward(tid);
        }
        let pf_ms = pf_start.elapsed().as_secs_f64() * 1000.0;

        // Generate (greedy)
        let mut generated: Vec<u32> = Vec::new();
        let gen_start = Instant::now();

        for _ in 0..40 {
            let next_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(id, _)| id as u32)
                .unwrap_or(0);

            if next_id == 151_643 || next_id == 151_645 || next_id == 151_644 {
                break;
            }
            generated.push(next_id);
            logits = model.forward(next_id);
        }
        let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let tps = if gen_ms > 0.0 {
            generated.len() as f64 / gen_ms * 1000.0
        } else {
            0.0
        };

        let output = tokenizer.decode(&generated, true).unwrap_or_default();

        println!("  A: {output}");
        println!(
            "  ({} tokens, prefill {pf_ms:.0}ms, gen {gen_ms:.0}ms, {tps:.2} tok/s)",
            generated.len()
        );

        // Quality checks
        let has_think = output.contains("<think");
        let has_tool = output.contains("arithmetic")
            || output.contains("vectordb")
            || output.contains("lakehouse")
            || output.contains("\"name\"");
        let has_json = output.contains('{') && output.contains('}');
        let is_blank = output.trim().is_empty();
        println!(
            "  [think={} tool={} json={} blank={}]",
            if has_think { "Y" } else { "n" },
            if has_tool { "Y" } else { "n" },
            if has_json { "Y" } else { "n" },
            if is_blank { "YES!" } else { "n" },
        );
    }

    println!("\n  Total: {:.1}s", start.elapsed().as_secs_f32());
}

// ---------------------------------------------------------------------------
// FP32 bin parser
// ---------------------------------------------------------------------------
fn parse_fp32_bin(data: &[u8]) -> HashMap<String, Vec<f32>> {
    let mut map = HashMap::new();
    let mut offset = 0;

    if data.len() < 4 {
        return map;
    }

    let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    for _ in 0..count {
        if offset + 4 > data.len() {
            break;
        }
        let name_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
        offset += name_len;
        let arr_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut arr = Vec::with_capacity(arr_len);
        for _ in 0..arr_len {
            let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            arr.push(val);
            offset += 4;
        }
        map.insert(name, arr);
    }
    map
}

// ---------------------------------------------------------------------------
// Build model from ATML archive + FP32 data
// ---------------------------------------------------------------------------
fn build_model_from_atml(
    archive: &ModelArchive,
    fp32: &HashMap<String, Vec<f32>>,
    config: &Llama3TernaryConfig,
) -> Llama3TernaryModel {
    // This creates the model by directly loading pre-quantized ternary weights.
    // We use the public from_parts constructor (which we need to add),
    // or reconstruct via the existing from_safetensors path.
    //
    // For now, we use a workaround: build the model struct manually
    // by accessing the archive layers in order.

    // The ATML contains layers in this order:
    // [0] output_proj
    // [1..197] layer.{0..27}.{q,k,v,o,gate,up,down}_proj

    let output_proj =
        ModelArchive::restore_weight(archive.get_layer(0).expect("missing output_proj"));

    let embedding = fp32.get("embedding").expect("missing embedding").clone();
    let output_norm = fp32
        .get("output_norm")
        .expect("missing output_norm")
        .clone();

    let proj_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    let mut layers_data = Vec::new();

    for layer_idx in 0..config.num_layers {
        let attn_norm = fp32
            .get(&format!("layer.{layer_idx}.attn_norm"))
            .expect("missing attn_norm")
            .clone();
        let ffn_norm = fp32
            .get(&format!("layer.{layer_idx}.ffn_norm"))
            .expect("missing ffn_norm")
            .clone();

        let mut proj_weights = Vec::new();
        for (j, _name) in proj_names.iter().enumerate() {
            let atml_idx = 1 + layer_idx * 7 + j;
            let tw = ModelArchive::restore_weight(
                archive
                    .get_layer(atml_idx)
                    .unwrap_or_else(|| panic!("missing layer {atml_idx}")),
            );
            proj_weights.push(tw);
        }

        layers_data.push((attn_norm, ffn_norm, proj_weights));
    }

    // Use the from_parts method
    Llama3TernaryModel::from_parts(
        config.clone(),
        embedding,
        output_norm,
        output_proj,
        layers_data,
    )
}
