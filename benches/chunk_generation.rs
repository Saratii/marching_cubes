#[path = "bench_util.rs"]
mod bench_util;

use criterion::{Criterion, criterion_group, criterion_main};

use marching_cubes::{
    constants::{
        SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_2D_PADDED, SAMPLES_PER_CHUNK_DIM,
        SAMPLES_PER_CHUNK_PADDED,
    },
    deformable_terrain::{
        chunk_generator::{
            calculate_chunk_start, chunk_contains_surface, compute_heightmap_gradients, downscale,
            fast_get_uniformity, fill_voxel_densities, generate_chunk_into_buffers,
            generate_noise_height_samples, generate_terrain_heights,
        },
        driver::{ChunkBuffers, LodBuffers},
        marching_cubes::mc::mc_mesh_generation,
        plugin::{NoiseHeightConfig, Uniformity},
    },
};
use std::{hint::black_box, time::Duration};

use crate::bench_util::find_chunk_with_surface;

fn benchmark_generate_chunk_into_buffers(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let fbm = NoiseHeightConfig::default();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    let uniformity = fast_get_uniformity(
        &chunk_buffers.heightmap,
        &chunk_buffers.dhdx,
        &chunk_buffers.dhdz,
        &chunk_start,
    );
    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("generate_chunk_into_buffers", |b| {
        b.iter(|| {
            black_box(generate_chunk_into_buffers(
                black_box(chunk_start),
                black_box(&mut chunk_buffers),
            ));
        })
    });
}

fn benchmark_generate_uniform_densities_cpu(c: &mut Criterion) {
    let fbm = NoiseHeightConfig::default();
    let chunk_start = calculate_chunk_start(&(0, 2000, 0));
    let mut chunk_buffers = ChunkBuffers::new();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    c.bench_function("generate_uniform_densities_cpu", |b| {
        b.iter(|| {
            black_box(generate_chunk_into_buffers(
                black_box(chunk_start),
                black_box(&mut chunk_buffers),
            ))
        })
    });
}

fn benchmark_marching_cubes(c: &mut Criterion) {
    let chunk = find_chunk_with_surface();
    let fbm = NoiseHeightConfig::default();
    let chunk_start = calculate_chunk_start(&chunk);
    let mut chunk_buffers = ChunkBuffers::new();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    generate_chunk_into_buffers(chunk_start, black_box(&mut chunk_buffers));
    c.bench_function("marching_cubes", |b| {
        b.iter(|| {
            black_box(mc_mesh_generation(
                black_box(&chunk_buffers.density),
                black_box(&chunk_buffers.material),
                black_box(SAMPLES_PER_CHUNK_DIM),
                false,
                black_box(&chunk_buffers.density),
            ));
        })
    });
}

fn benchmark_compute_heightmap_gradients(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let fbm = NoiseHeightConfig::default();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let height_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    let mut dhdx_buffer = [0.0; SAMPLES_PER_CHUNK_2D_PADDED];
    let mut dhdz_buffer = [0.0; SAMPLES_PER_CHUNK_2D_PADDED];
    c.bench_function("compute_heightmap_gradients", |b| {
        b.iter(|| {
            black_box(compute_heightmap_gradients(
                black_box(&mut dhdx_buffer),
                black_box(&mut dhdz_buffer),
                black_box(&height_samples),
            ));
        })
    });
}

fn bench_downscale(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let mut chunk_buffers = ChunkBuffers::new();
    let mut lod_buffers = LodBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = NoiseHeightConfig::default();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let uniformity = fast_get_uniformity(
        &chunk_buffers.heightmap,
        &chunk_buffers.dhdx,
        &chunk_buffers.dhdz,
        &chunk_start,
    );
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.benchmark_group("downscale")
        .measurement_time(Duration::from_secs(10))
        .bench_function("downscale", |b| {
            b.iter(|| {
                black_box(downscale(
                    black_box(&chunk_buffers.density),
                    black_box(&chunk_buffers.material),
                    black_box(&mut lod_buffers.density_r1),
                    black_box(&mut lod_buffers.material_r1),
                    black_box(32),
                ));
            })
        });
}

fn bench_chunk_contains_surface_full(c: &mut Criterion) {
    let density_buffer = [-10; SAMPLES_PER_CHUNK_PADDED]; //no surface is the common and worst case
    c.bench_function("chunk_contains_surface_full", |b| {
        b.iter(|| {
            black_box(chunk_contains_surface(black_box(&density_buffer)));
        })
    });
}

fn bench_chunk_contains_surface_r1(c: &mut Criterion) {
    let density_r1 = [-10; SAMPLES_PER_CHUNK / 2_usize.pow(3)]; //no surface is the common and worst case
    c.bench_function("chunk_contains_surface_r1", |b| {
        b.iter(|| {
            black_box(chunk_contains_surface(black_box(&density_r1)));
        })
    });
}

fn bench_chunk_contains_surface_r5(c: &mut Criterion) {
    let density_r5 = [-10; SAMPLES_PER_CHUNK / 32_usize.pow(3)]; //no surface is the common and worst case
    c.bench_function("chunk_contains_surface_r5", |b| {
        b.iter(|| {
            black_box(chunk_contains_surface(black_box(&density_r5)));
        })
    });
}

fn bench_generate_noise_height_samples(c: &mut Criterion) {
    let chunk_start_x = -894.;
    let chunk_start_z = 1242.;
    let fbm = NoiseHeightConfig::default();
    c.bench_function("generate_noise_height_samples", |b| {
        b.iter(|| {
            black_box(generate_noise_height_samples(
                black_box(chunk_start_x),
                black_box(chunk_start_z),
                black_box(&fbm),
            ));
        })
    });
}

fn bench_generate_terrain_heights(c: &mut Criterion) {
    let mut heightmap_buffer = [0.0; SAMPLES_PER_CHUNK_2D_PADDED];
    let chunk_start_x = -894.;
    let chunk_start_z = 1242.;
    let fbm = NoiseHeightConfig::default();
    let noise_samples = generate_noise_height_samples(chunk_start_x, chunk_start_z, &fbm);
    c.bench_function("generate_terrain_heights", |b| {
        b.iter(|| {
            black_box(generate_terrain_heights(
                black_box(&mut heightmap_buffer),
                black_box(&noise_samples),
            ));
        })
    });
}

fn bench_fill_voxel_densities(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let fbm = NoiseHeightConfig::default();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let uniformity = fast_get_uniformity(
        &chunk_buffers.heightmap,
        &chunk_buffers.dhdx,
        &chunk_buffers.dhdz,
        &chunk_start,
    );
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("fill_voxel_densities", |b| {
        b.iter(|| {
            black_box(fill_voxel_densities(&mut chunk_buffers, &chunk_start));
        })
    });
}

fn bench_fast_get_uniformity_uniform(c: &mut Criterion) {
    let chunk_coord = (0, 2000, 0);
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = NoiseHeightConfig::default();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let uniformity = fast_get_uniformity(
        &chunk_buffers.heightmap,
        &chunk_buffers.dhdx,
        &chunk_buffers.dhdz,
        &chunk_start,
    );
    assert_eq!(uniformity, Uniformity::Air);
    c.bench_function("fast_get_uniformity_uniform", |b| {
        b.iter(|| {
            black_box(fast_get_uniformity(
                black_box(&chunk_buffers.heightmap),
                black_box(&chunk_buffers.dhdx),
                black_box(&chunk_buffers.dhdz),
                black_box(&chunk_start),
            ));
        })
    });
}

fn bench_fast_get_uniformity_non_uniform(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = NoiseHeightConfig::default();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let uniformity = fast_get_uniformity(
        &chunk_buffers.heightmap,
        &chunk_buffers.dhdx,
        &chunk_buffers.dhdz,
        &chunk_start,
    );
    assert_eq!(uniformity, Uniformity::NonUniform);
    c.bench_function("fast_get_uniformity_non_uniform", |b| {
        b.iter(|| {
            black_box(fast_get_uniformity(
                black_box(&chunk_buffers.heightmap),
                black_box(&chunk_buffers.dhdx),
                black_box(&chunk_buffers.dhdz),
                black_box(&chunk_start),
            ));
        })
    });
}

criterion_group!(
    benches,
    benchmark_generate_chunk_into_buffers,
    benchmark_marching_cubes,
    benchmark_generate_uniform_densities_cpu,
    benchmark_compute_heightmap_gradients,
    bench_downscale,
    bench_chunk_contains_surface_full,
    bench_chunk_contains_surface_r1,
    bench_chunk_contains_surface_r5,
    bench_generate_noise_height_samples,
    bench_generate_terrain_heights,
    bench_fill_voxel_densities,
    bench_fast_get_uniformity_uniform,
    bench_fast_get_uniformity_non_uniform,
);
criterion_main!(benches);

//cargo bench --bench chunk_generation -- deserialize_chunk_data

//cargo build --bench chunk_generation -r

//perf record ./target/release/deps/chunk_generation-6711b8d7beb362fa.exe --bench generate_densities_single_chunk
