use bevy::math::Vec3;
use criterion::{Criterion, criterion_group, criterion_main};

use marching_cubes::{
    conversions::{chunk_coord_to_cluster_coord, world_pos_to_chunk_coord},
    marching_cubes::mc::mc_mesh_generation,
    terrain::{
        chunk_compute_pipeline::GpuTerrainGenerator,
        chunk_generator::{
            calculate_chunk_start, chunk_contains_surface, generate_terrain_heights, get_fbm,
        },
        heightmap_compute_pipeline::GpuHeightmapGenerator,
        terrain::{
            CLUSTER_SIZE, HALF_CHUNK, HEIGHT_MAP_GRID_SIZE, SAMPLES_PER_CHUNK,
            SAMPLES_PER_CHUNK_DIM, generate_chunk_into_buffers,
        },
    },
};
use std::{collections::HashMap, hint::black_box};

fn benchmark_generate_densities_cpu(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let fbm = get_fbm();
    let mut densities_buffer = [0; SAMPLES_PER_CHUNK];
    let mut materials_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("generate_densities_cpu", |b| {
        b.iter(|| {
            let chunk_start = calculate_chunk_start(&chunk_coord);
            black_box(generate_chunk_into_buffers(
                black_box(&fbm),
                black_box(chunk_start),
                black_box(&mut densities_buffer),
                black_box(&mut materials_buffer),
                black_box(&mut heightmap_buffer),
                SAMPLES_PER_CHUNK_DIM,
            ));
        })
    });
}

fn benchmark_generate_uniform_densities_cpu(c: &mut Criterion) {
    let fbm = get_fbm();
    let mut densities_buffer = [0; SAMPLES_PER_CHUNK];
    let mut materials_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("generate_uniform_densities_cpu", |b| {
        b.iter(|| {
            let chunk_start = calculate_chunk_start(&(0, 2000, 0));
            black_box(generate_chunk_into_buffers(
                black_box(&fbm),
                black_box(chunk_start),
                black_box(&mut densities_buffer),
                black_box(&mut materials_buffer),
                black_box(&mut heightmap_buffer),
                SAMPLES_PER_CHUNK_DIM,
            ))
        })
    });
}

fn benchmark_generate_densities_gpu(c: &mut Criterion) {
    let gpu_generator = GpuTerrainGenerator::new();
    let chunk_coord = find_chunk_with_surface();
    c.bench_function("generate_densities_gpu", |b| {
        b.iter(|| {
            let (densities, materials, is_uniform) =
                gpu_generator.generate_densities(black_box(&chunk_coord));
            black_box((densities, materials, is_uniform));
        })
    });
}

fn benchmark_marching_cubes(c: &mut Criterion) {
    let chunk = find_chunk_with_surface();
    let fbm = get_fbm();
    let chunk_start = calculate_chunk_start(&chunk);
    let mut densities_buffer = [0; SAMPLES_PER_CHUNK];
    let mut materials_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    generate_chunk_into_buffers(
        &fbm,
        chunk_start,
        &mut densities_buffer,
        &mut materials_buffer,
        &mut heightmap_buffer,
        SAMPLES_PER_CHUNK_DIM,
    );
    c.bench_function("marching_cubes", |b| {
        b.iter(|| {
            black_box(mc_mesh_generation(
                black_box(&densities_buffer),
                black_box(&materials_buffer),
                black_box(SAMPLES_PER_CHUNK_DIM),
                black_box(HALF_CHUNK),
            ));
        })
    });
}

fn benchmark_heightmap_single_chunk_cpu(c: &mut Criterion) {
    let chunk_coord = (0, 0, 0);
    let fbm = get_fbm();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    let chunk_start = calculate_chunk_start(&chunk_coord);
    c.bench_function("heightmap_single_cpu", |b| {
        b.iter(|| {
            black_box(generate_terrain_heights(
                black_box(chunk_start.x),
                black_box(chunk_start.z),
                black_box(&fbm),
                black_box(&mut heightmap_buffer),
                SAMPLES_PER_CHUNK_DIM,
            ));
        })
    });
}

fn benchmark_heightmap_single_chunk_gpu(c: &mut Criterion) {
    let chunk_coord = (0, 0, 0);
    let gpu_generator = GpuHeightmapGenerator::new();
    c.bench_function("heightmap_single_gpu", |b| {
        b.iter(|| {
            black_box(gpu_generator.generate_heightmap(black_box(&chunk_coord)));
        })
    });
}

fn benchmark_cluster_heightmap_cpu(c: &mut Criterion) {
    let fbm = get_fbm();
    let chunk = world_pos_to_chunk_coord(&Vec3::ZERO);
    let cluster = chunk_coord_to_cluster_coord(&chunk);
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("cluster_heightmap_cpu", |b| {
        b.iter(|| {
            let mut results = HashMap::new();
            for x in cluster.0..cluster.0 + CLUSTER_SIZE as i16 {
                for z in cluster.2..cluster.2 + CLUSTER_SIZE as i16 {
                    let chunk_coord = (x, cluster.1, z);
                    let chunk_start = calculate_chunk_start(&chunk_coord);
                    let heights = generate_terrain_heights(
                        black_box(chunk_start.x),
                        black_box(chunk_start.z),
                        black_box(&fbm),
                        black_box(&mut heightmap_buffer),
                        SAMPLES_PER_CHUNK_DIM,
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

fn find_chunk_with_surface() -> (i16, i16, i16) {
    let mut densities_buffer = [0; SAMPLES_PER_CHUNK];
    let mut materials_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    let fbm = get_fbm();
    for chunk_y in -100..100 {
        let chunk_coord = (0, chunk_y, 0);
        let chunk_start = calculate_chunk_start(&chunk_coord);
        generate_chunk_into_buffers(
            &fbm,
            chunk_start,
            &mut densities_buffer,
            &mut materials_buffer,
            &mut heightmap_buffer,
            SAMPLES_PER_CHUNK_DIM,
        );
        if chunk_contains_surface(&densities_buffer) {
            return chunk_coord;
        }
    }
    panic!("No chunk with surface found in the tested range");
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
