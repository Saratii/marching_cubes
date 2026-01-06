use bevy::math::Vec3;
use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::{
    conversions::{chunk_coord_to_cluster_coord, world_pos_to_chunk_coord},
    marching_cubes::mc::mc_mesh_generation,
    terrain::{
        chunk_compute_pipeline::GpuTerrainGenerator,
        chunk_generator::{
            calculate_chunk_start, chunk_contains_surface, generate_densities,
            generate_terrain_heights, sample_fbm,
        },
        heightmap_compute_pipeline::GpuHeightmapGenerator,
        terrain::{CLUSTER_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk},
    },
};
use std::{collections::HashMap, hint::black_box};

fn benchmark_generate_densities_cpu(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    c.bench_function("generate_densities_cpu", |b| {
        b.iter(|| {
            let chunk_start = calculate_chunk_start(&(0, 2, 0));
            let first_sample_reuse = sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
            black_box(generate_densities(
                black_box(&noise_function),
                black_box(first_sample_reuse),
                black_box(chunk_start),
            ))
        })
    });
}

fn benchmark_generate_uniform_densities_cpu(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    c.bench_function("generate_uniform_densities_cpu", |b| {
        b.iter(|| {
            let chunk_start = calculate_chunk_start(&(0, 2000, 0));
            let first_sample_reuse = sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
            black_box(generate_densities(
                black_box(&noise_function),
                black_box(first_sample_reuse),
                black_box(chunk_start),
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
    let chunk_start = calculate_chunk_start(&chunk);
    let first_sample_reuse = sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
    let chunk = TerrainChunk::new(&noise_function, chunk_start, first_sample_reuse);
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
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let first_sample_reuse = sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
    let chunk = TerrainChunk::new(&noise_function, chunk_start, first_sample_reuse);
    assert!(
        chunk_contains_surface(&chunk),
        "Chunk at {:?} should contain a surface",
        chunk_coord
    );
    let chunk_start = calculate_chunk_start(&chunk_coord);
    c.bench_function("heightmap_single_cpu", |b| {
        b.iter(|| {
            let first_sample_reuse = sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
            black_box(generate_terrain_heights(
                black_box(chunk_start.x),
                black_box(chunk_start.z),
                black_box(&noise_function),
                black_box(first_sample_reuse),
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

fn benchmark_cluster_heightmap_cpu(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let chunk = world_pos_to_chunk_coord(&Vec3::ZERO);
    let cluster = chunk_coord_to_cluster_coord(&chunk);
    c.bench_function("cluster_heightmap_cpu", |b| {
        b.iter(|| {
            let mut results = HashMap::new();
            for x in cluster.0..cluster.0 + CLUSTER_SIZE as i16 {
                for z in cluster.2..cluster.2 + CLUSTER_SIZE as i16 {
                    let chunk_coord = (x, cluster.1, z);
                    let chunk_start = calculate_chunk_start(&chunk_coord);
                    let first_sample_reuse =
                        sample_fbm(&noise_function, chunk_start.x, chunk_start.z);
                    let heights = generate_terrain_heights(
                        black_box(chunk_start.x),
                        black_box(chunk_start.z),
                        black_box(&noise_function),
                        black_box(first_sample_reuse),
                    );
                    results.insert((chunk_coord.0, chunk_coord.1), heights);
                }
            }
            black_box(results);
        })
    });
}

fn benchmark_cluster_heightmap_gpu(c: &mut Criterion) {
    let gpu_generator = GpuHeightmapGenerator::new();
    let chunk = world_pos_to_chunk_coord(&Vec3::ZERO);
    let cluster = chunk_coord_to_cluster_coord(&chunk);
    c.bench_function("cluster_heightmap_gpu", |b| {
        b.iter(|| {
            black_box(
                gpu_generator
                    .generate_cluster(black_box(cluster.0 as i32), black_box(cluster.2 as i32)),
            );
        })
    });
}

fn benchmark_batch_cluster_heightmaps_gpu(c: &mut Criterion) {
    let gpu_generator = GpuHeightmapGenerator::new();
    let mut cluster_coords = Vec::with_capacity(100 * 100);
    for z in -50..50 {
        for x in -50..50 {
            cluster_coords.push((x, z));
        }
    }

    c.bench_function("batch_cluster_heightmaps_gpu", |b| {
        b.iter(|| {
            black_box(gpu_generator.generate_batch_clusters(black_box(&cluster_coords)));
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
    benchmark_cluster_heightmap_cpu,
    benchmark_cluster_heightmap_gpu,
    benchmark_generate_uniform_densities_cpu,
    benchmark_batch_cluster_heightmaps_gpu,
);
criterion_main!(benches);

//cargo bench --bench chunk_generation -- deserialize_chunk_data

//cargo build --bench chunk_generation -r

//perf record ./target/release/deps/chunk_generation-6711b8d7beb362fa.exe --bench generate_densities_single_chunk
