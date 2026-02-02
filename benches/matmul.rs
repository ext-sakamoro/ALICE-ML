//! Benchmarks for Ternary MatMul (Supernova Edition)
//!
//! Tests DPS (Destination Passing Style) kernels with zero allocation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use alice_ml::{
    TernaryWeight, TernaryWeightKernel,
    ternary_matvec, ternary_matvec_kernel,
    quantize_to_ternary,
};

fn bench_ternary_matvec_dps(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_matvec_dps");

    for size in [64, 256, 512, 1024].iter() {
        // Create input
        let input: Vec<f32> = (0..*size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        // Create ternary weights
        let weight_data: Vec<i8> = (0..*size * *size)
            .map(|i| ((i % 3) as i8 - 1))
            .collect();
        let weights = TernaryWeight::from_ternary(&weight_data, *size, *size);

        // Pre-allocate output (zero allocation during bench)
        let mut output = vec![0.0f32; *size];

        group.bench_with_input(
            BenchmarkId::new("packed", size),
            size,
            |b, _| {
                b.iter(|| {
                    ternary_matvec(black_box(&input), &weights, &mut output);
                    black_box(&output)
                })
            },
        );
    }

    group.finish();
}

fn bench_ternary_kernel_dps(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_kernel_dps");

    for size in [64, 256, 512, 1024].iter() {
        let input: Vec<f32> = (0..*size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let weight_data: Vec<i8> = (0..*size * *size)
            .map(|i| ((i % 3) as i8 - 1))
            .collect();
        let kernel = TernaryWeightKernel::from_ternary(&weight_data, *size, *size);

        let mut output = vec![0.0f32; *size];

        group.bench_with_input(
            BenchmarkId::new("bitparallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    ternary_matvec_kernel(black_box(&input), &kernel, &mut output);
                    black_box(&output)
                })
            },
        );
    }

    group.finish();
}

fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");

    for size in [256, 512, 1024].iter() {
        let weights: Vec<f32> = (0..*size * *size)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("size", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(quantize_to_ternary(&weights, *size, *size))
                })
            },
        );
    }

    group.finish();
}

fn bench_fp32_matvec_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp32_matvec_baseline");

    for size in [64, 256, 512, 1024].iter() {
        let input: Vec<f32> = (0..*size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let weights: Vec<f32> = (0..*size * *size)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();

        let mut output = vec![0.0f32; *size];

        group.bench_with_input(
            BenchmarkId::new("size", size),
            size,
            |b, size| {
                b.iter(|| {
                    // Naive FP32 matmul for comparison
                    for i in 0..*size {
                        let mut sum = 0.0f32;
                        for j in 0..*size {
                            sum += weights[i * size + j] * input[j];
                        }
                        output[i] = sum;
                    }
                    black_box(&output)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ternary_matvec_dps,
    bench_ternary_kernel_dps,
    bench_quantization,
    bench_fp32_matvec_baseline
);
criterion_main!(benches);
