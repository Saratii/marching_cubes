use crate::bench_util::find_chunk_with_surface;
use criterion::{Criterion, criterion_group, criterion_main};
use crossbeam_channel::unbounded;
use marching_cubes::deformable_terrain::chunk_generator::{
    calculate_chunk_start, chunk_contains_surface, compute_heightmap_gradients,
    fast_get_uniformity, generate_chunk_into_buffers, generate_noise_height_samples,
    generate_terrain_heights,
};
use marching_cubes::deformable_terrain::driver::{
    ChunkBuffers, ChunkSpawnResult, ClusterRequest, FullLodMode, LoadStateTransition, LodBuffers,
    build_full_mesh_and_spawn, lod_resolve_has_surface, try_load_chunk,
};
use marching_cubes::deformable_terrain::file_loader::load_chunk_index_map;
use marching_cubes::deformable_terrain::plugin::{NoiseHeightConfig, Uniformity};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
#[cfg(windows)]
use std::fs::File;
use std::fs::OpenOptions;
use std::hint::black_box;

#[path = "bench_util.rs"]
mod bench_util;

fn benchmark_build_full_mesh_and_spawn_with_collider(c: &mut Criterion) {
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
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),                                //shouldnt matter
        distance_squared: 0.0,                              //shouldnt matter
        load_state_transition: LoadStateTransition::ToFull, //shouldnt matter
        prev_has_entity: None,
        prev_in_simulation_radius: false,
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("build_full_mesh_and_spawn_with_collider", |b| {
        b.iter(|| {
            build_full_mesh_and_spawn(
                black_box(&chunk_buffers.density),
                black_box(&chunk_buffers.material),
                black_box(chunk_coord),
                black_box(&cluster_request),
                black_box(0),
                black_box(&chunk_spawn_sender),
                black_box(FullLodMode::WithCollider),
            );
        })
    });
}

fn benchmark_build_full_mesh_and_spawn_no_collider(c: &mut Criterion) {
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
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),                                //shouldnt matter
        distance_squared: 0.0,                              //shouldnt matter
        load_state_transition: LoadStateTransition::ToFull, //shouldnt matter
        prev_has_entity: None,
        prev_in_simulation_radius: false,
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("build_full_mesh_and_spawn_no_collider", |b| {
        b.iter(|| {
            black_box(build_full_mesh_and_spawn(
                black_box(&chunk_buffers.density),
                black_box(&chunk_buffers.material),
                black_box(chunk_coord),
                black_box(&cluster_request),
                black_box(0),
                black_box(&chunk_spawn_sender),
                black_box(FullLodMode::NoCollider),
            ));
        })
    });
}

fn bench_resolve_has_surface_lod5(c: &mut Criterion) {
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
    let _uniformity = generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToLod5,
        prev_has_entity: None, //shouldnt matter
        prev_in_simulation_radius: false,
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_lod5", |b| {
        b.iter(|| {
            black_box(lod_resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(0),
                black_box(&chunk_spawn_sender),
            ));
        })
    });
}

fn bench_resolve_has_surface_lod1(c: &mut Criterion) {
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
    let _uniformity = generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToLod1,
        prev_has_entity: None, //shouldnt matter
        prev_in_simulation_radius: false,
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_lod1", |b| {
        b.iter(|| {
            black_box(lod_resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(0),
                black_box(&chunk_spawn_sender),
            ));
        })
    });
}

fn bench_resolve_has_surface_full_collider(c: &mut Criterion) {
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
    let _uniformity = generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToFullWithCollider,
        prev_has_entity: None, //shouldnt matter
        prev_in_simulation_radius: false,
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_full_collider", |b| {
        b.iter(|| {
            black_box(lod_resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(0),
                black_box(&chunk_spawn_sender),
            ));
        })
    });
}

fn bench_try_load_chunk_fail(c: &mut Criterion) {
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
    let index_map_delta = RwLock::new(FxHashMap::default());
    let index_map_read = FxHashMap::default();
    #[cfg(windows)]
    let mut chunk_data_file_read = File::open("NUL").unwrap();
    #[cfg(not(windows))]
    let mut chunk_data_file_read = File::open("/dev/null").unwrap();
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("try_load_chunk_fail", |b| {
        b.iter(|| {
            black_box(try_load_chunk(
                black_box(chunk_coord),
                black_box(&index_map_read),
                black_box(&index_map_delta),
                black_box(&mut chunk_data_file_read),
                black_box(&mut chunk_buffers),
            ));
        })
    });
}

fn bench_try_load_chunk_success(c: &mut Criterion) {
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
    let index_map_delta = RwLock::new(FxHashMap::default());
    let mut chunk_index_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("benches/bench_data/chunk_index_data.txt")
        .unwrap();
    let index_map_read = load_chunk_index_map(&mut chunk_index_file);
    let mut chunk_data_file_read = File::open("benches/bench_data/chunk_data.txt").unwrap();
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("try_load_chunk_success", |b| {
        b.iter(|| {
            black_box(try_load_chunk(
                black_box(chunk_coord),
                black_box(&index_map_read),
                black_box(&index_map_delta),
                black_box(&mut chunk_data_file_read),
                black_box(&mut chunk_buffers),
            ));
        })
    });
}

criterion_group!(
    benches,
    benchmark_build_full_mesh_and_spawn_with_collider,
    benchmark_build_full_mesh_and_spawn_no_collider,
    bench_resolve_has_surface_lod5,
    bench_resolve_has_surface_lod1,
    bench_resolve_has_surface_full_collider,
    bench_try_load_chunk_fail,
    bench_try_load_chunk_success,
);
criterion_main!(benches);

//cargo bench --bench driver_bench -- build_full_mesh_and_spawn_with_collider
