use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::{
    marching_cubes::mc::mc_mesh_generation,
    terrain::{
        chunk_compute_pipeline::GpuTerrainGenerator, chunk_generator::{
            calculate_chunk_start, chunk_contains_surface, generate_densities,
            generate_terrain_heights,
        }, heightmap_compute_pipeline::GpuHeightmapGenerator, terrain::{HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk}
    },
};
use std::hint::black_box;

fn benchmark_generate_densities_cpu(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    c.bench_function("generate_densities_cpu", |b| {
        b.iter(|| {
            black_box(generate_densities(
                black_box(&(0, 2, 0)),
                black_box(&noise_function),
            ))
        })
    });
}

fn benchmark_generate_densities_gpu(c: &mut Criterion) {
    let gpu_generator = GpuTerrainGenerator::new();
    let chunk_coord = (0, 0, 0);
    c.bench_function("generate_densities_gpu", |b| {
        b.iter(|| {
            let (densities, materials, is_uniform) =
                gpu_generator.generate_densities(black_box(&chunk_coord));
            black_box((densities, materials, is_uniform));
        })
    });
}

fn benchmark_marching_cubes(c: &mut Criterion) {
    let chunk = (0, 2, 0);
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let chunk = TerrainChunk::new(chunk, &noise_function);
    assert!(
        chunk_contains_surface(&chunk),
        "Chunk at {:?} should contain a surface",
        chunk
    );
    c.bench_function("marching_cubes", |b| {
        b.iter(|| {
            black_box(mc_mesh_generation(
                black_box(&chunk.densities),
                black_box(&chunk.materials),
                black_box(SAMPLES_PER_CHUNK_DIM),
                black_box(HALF_CHUNK),
            ));
        })
    });
}

fn benchmark_heightmap_single_chunk_cpu(c: &mut Criterion) {
    let chunk_coord = (0, 2, 0);
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let chunk = TerrainChunk::new(chunk_coord, &noise_function);
    assert!(
        chunk_contains_surface(&chunk),
        "Chunk at {:?} should contain a surface",
        chunk_coord
    );
    let chunk_start = calculate_chunk_start(&chunk_coord);
    c.bench_function("heightmap_single_cpu", |b| {
        b.iter(|| {
            black_box(generate_terrain_heights(
                black_box(&chunk_start),
                black_box(&noise_function),
            ));
        })
    });
}

fn benchmark_heightmap_single_chunk_gpu(c: &mut Criterion) {
    let chunk_coord = (0, 2, 0);
    let gpu_generator = GpuHeightmapGenerator::new();
    c.bench_function("heightmap_single_gpu", |b| {
        b.iter(|| {
            black_box(gpu_generator.generate_heightmap(black_box(&chunk_coord)));
        })
    });
}

fn benchmark_heightmap_region_cpu(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let lower = (0, 0, 0);
    let upper = (10, 0, 10);
    c.bench_function("heightmap_region_cpu", |b| {
        b.iter(|| {
            let mut results = std::collections::HashMap::new();
            for x in lower.0..=upper.0 {
                for z in lower.2..=upper.2 {
                    let chunk_coord = (x, lower.1, z);
                    let chunk_start = calculate_chunk_start(&chunk_coord);
                    let heights = generate_terrain_heights(
                        black_box(&chunk_start),
                        black_box(&noise_function),
                    );
                    results.insert(chunk_coord, heights);
                }
            }
            black_box(results);
        })
    });
}

fn benchmark_heightmap_region_gpu(c: &mut Criterion) {
    let gpu_generator = GpuHeightmapGenerator::new();
    let lower = (0, 0, 0);
    let upper = (10, 0, 10);
    c.bench_function("heightmap_region_gpu", |b| {
        b.iter(|| {
            black_box(gpu_generator.generate_region(black_box(lower), black_box(upper)));
        })
    });
}

criterion_group!(
    benches,
    benchmark_generate_densities_cpu,
    benchmark_marching_cubes,
    benchmark_generate_densities_gpu,
    benchmark_heightmap_single_chunk_cpu,
    benchmark_heightmap_single_chunk_gpu,
    benchmark_heightmap_region_cpu,
    benchmark_heightmap_region_gpu,
);
criterion_main!(benches);

//cargo bench --bench chunk_generation -- deserialize_chunk_data

//cargo build --bench chunk_generation -r

//perf record ./target/release/deps/chunk_generation-6711b8d7beb362fa.exe --bench generate_densities_single_chunk
