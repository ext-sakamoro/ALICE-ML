//! ELYZA-JP-8B Ternary (1.58-bit) inference example.
//!
//! Two modes:
//! 1. Quantize from safetensors → ATML (one-time):
//!    cargo run --example elyza_ternary --features safetensors,neon -- \
//!      --quantize --input models/elyza-8b/ --output models/elyza-8b-ternary.atml
//!
//! 2. Run inference from ATML:
//!    cargo run --example elyza_ternary --features safetensors,neon -- \
//!      --model models/elyza-8b-ternary.atml \
//!      --prompt "日本の首都はどこですか？"
//!
//! Download model:
//!   huggingface-cli download elyza/Llama-3-ELYZA-JP-8B --local-dir models/elyza-8b/

#[cfg(feature = "safetensors")]
fn main() {
    use alice_ml::llama3_ternary::{Llama3TernaryConfig, Llama3TernaryModel};
    use alice_ml::safetensors::SafetensorsFile;
    use std::env;
    use std::fs;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--quantize") {
        // ── Quantize mode ───────────────────────────────────────────────
        let input_dir = args
            .iter()
            .position(|a| a == "--input")
            .and_then(|i| args.get(i + 1))
            .expect("Usage: --quantize --input <dir> --output <file.atml>");

        let output_path = args
            .iter()
            .position(|a| a == "--output")
            .and_then(|i| args.get(i + 1))
            .expect("Usage: --quantize --input <dir> --output <file.atml>");

        println!("Quantizing model from: {input_dir}");
        let start = Instant::now();

        // Find safetensors files
        let mut shard_files: Vec<String> = Vec::new();
        for entry in fs::read_dir(input_dir).expect("Cannot read input dir") {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") {
                shard_files.push(entry.path().to_string_lossy().to_string());
            }
        }
        shard_files.sort();

        if shard_files.is_empty() {
            eprintln!("No .safetensors files found in {input_dir}");
            std::process::exit(1);
        }

        println!("  Found {} shard(s)", shard_files.len());

        // For single shard models
        if shard_files.len() == 1 {
            let data = fs::read(&shard_files[0]).expect("Failed to read safetensors");
            let sf = SafetensorsFile::parse(&data).expect("Failed to parse safetensors");

            println!("  Tensors in file: {}", sf.len());

            let config = Llama3TernaryConfig::llama3_8b();
            let model = Llama3TernaryModel::from_safetensors(&sf, config)
                .expect("Failed to quantize model");

            let mem_mb = model.memory_bytes() as f64 / 1e6;
            println!("  Ternary model size: {mem_mb:.1} MB");
            println!("  Quantization report: {:?}", model.quant_stats);

            let atml_data = model.save_atml();
            fs::write(output_path, &atml_data).expect("Failed to write ATML");
            println!(
                "  Saved ATML: {} bytes ({:.1} MB)",
                atml_data.len(),
                atml_data.len() as f64 / 1e6
            );
        } else {
            eprintln!("Multi-shard quantization not yet implemented. Use single-shard model or merge first.");
            std::process::exit(1);
        }

        let elapsed = start.elapsed().as_secs();
        println!("  Total quantization time: {elapsed}s");
    } else {
        // ── Inference mode (placeholder) ────────────────────────────────
        println!("Ternary inference mode - ATML loading not yet fully implemented.");
        println!("Use --quantize first to create the ATML file.");
        println!("Full inference pipeline requires tokenizer integration (WIP).");
    }
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    eprintln!("This example requires the 'safetensors' feature:");
    eprintln!("  cargo run --example elyza_ternary --features safetensors");
}
