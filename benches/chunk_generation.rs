use criterion::{Criterion, criterion_group, criterion_main};
use marching_cubes::terrain_generation::generate_densities;
use noise::{Fbm, MultiFractal, Simplex};
use std::hint::black_box;

fn benchmark_generate_densities(c: &mut Criterion) {
    const NOISE_FREQUENCY: f64 = 0.02; // Frequency of the noise
    const NOISE_OCTAVES: usize = 3; // Number of octaves for the noise
    const NOISE_LACUNARITY: f64 = 2.1; // Lacunarity for the noise
    const NOISE_PERSISTENCE: f64 = 0.4; // Persistence for the noise
    const NOISE_SEED: u32 = 100; // Seed for the noise function
    let noise_function = Fbm::<Simplex>::new(NOISE_SEED)
        .set_frequency(NOISE_FREQUENCY)
        .set_octaves(NOISE_OCTAVES)
        .set_lacunarity(NOISE_LACUNARITY)
        .set_persistence(NOISE_PERSISTENCE);
    c.bench_function("generate_densities_single_chunk", |b| {
        b.iter(|| generate_densities(black_box(&(0, 0, 0)), black_box(&noise_function)))
    });
}

criterion_group!(benches, benchmark_generate_densities,);
criterion_main!(benches);
