use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::terrain::chunk_generator::generate_densities;
use std::hint::black_box;

fn benchmark_generate_densities(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    c.bench_function("generate_densities_single_chunk", |b| {
        b.iter(|| generate_densities(black_box(&(0, 0, 0)), black_box(&noise_function)))
    });
}

criterion_group!(benches, benchmark_generate_densities,);
criterion_main!(benches);

//cargo bench --bench chunk_generation -- deserialize_chunk_data
