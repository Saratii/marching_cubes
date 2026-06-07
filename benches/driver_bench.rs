#[cfg(windows)]
use std::fs::File;
use std::fs::OpenOptions;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use crossbeam_channel::unbounded;
use marching_cubes::data_loader::file_loader::load_chunk_index_map;
use marching_cubes::{
    data_loader::{
        column_range_map::ColumnRangeMap,
        driver::{
            ChunkBuffers, ChunkSpawnResult, ClusterRequest, FullLodMode, LoadStateTransition,
            LodBuffers, build_full_mesh_and_spawn, resolve_has_surface, resolve_uniformity,
        },
    },
    terrain::{
        chunk_generator::{
            calculate_chunk_start, chunk_contains_surface, generate_chunk_into_buffers, get_fbm,
        },
        terrain::Uniformity,
    },
};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::bench_util::find_chunk_with_surface;

#[path = "bench_util.rs"]
mod bench_util;

fn benchmark_build_full_mesh_and_spawn_with_collider(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = get_fbm();
    let uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),                                //shouldnt matter
        distance_squared: 0.0,                              //shouldnt matter
        load_state_transition: LoadStateTransition::ToFull, //shouldnt matter
        prev_has_entity: None,
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
    let fbm = get_fbm();
    let uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),                                //shouldnt matter
        distance_squared: 0.0,                              //shouldnt matter
        load_state_transition: LoadStateTransition::ToFull, //shouldnt matter
        prev_has_entity: None,
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
    let fbm = get_fbm();
    let _uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToLod5,
        prev_has_entity: None, //shouldnt matter
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_lod5", |b| {
        b.iter(|| {
            black_box(resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(&0),
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
    let fbm = get_fbm();
    let _uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToLod1,
        prev_has_entity: None, //shouldnt matter
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_lod1", |b| {
        b.iter(|| {
            black_box(resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(&0),
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
    let fbm = get_fbm();
    let _uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let cluster_request = ClusterRequest {
        position: (0, 0, 0),   //shouldnt matter
        distance_squared: 0.0, //shouldnt matter
        load_state_transition: LoadStateTransition::ToFullWithCollider,
        prev_has_entity: None, //shouldnt matter
    };
    let (chunk_spawn_sender, _chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    c.bench_function("resolve_has_surface_full_collider", |b| {
        b.iter(|| {
            black_box(resolve_has_surface(
                black_box(&cluster_request),
                black_box(&chunk_buffers),
                black_box(&mut lod_buffers),
                black_box((0, 0, 0)),
                black_box(&0),
                black_box(&chunk_spawn_sender),
            ));
        })
    });
}

fn bench_resolve_uniformity_from_generation(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = get_fbm();
    let uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let column_range_map = ColumnRangeMap::new();
    let index_map_delta = RwLock::new(FxHashMap::default());
    let index_map_read = FxHashMap::default();
    #[cfg(windows)]
    let mut chunk_data_file_read = File::open("NUL").unwrap();
    #[cfg(not(windows))]
    let mut chunk_data_file_read = File::open("/dev/null").unwrap();
    assert_eq!(uniformity, Uniformity::NonUniform);
    assert!(chunk_contains_surface(&chunk_buffers.density));
    c.bench_function("resolve_uniformity_from_generation", |b| {
        b.iter(|| {
            black_box(resolve_uniformity(
                black_box(chunk_coord),
                black_box(&column_range_map),
                black_box(&index_map_read),
                black_box(&index_map_delta),
                black_box(&mut chunk_data_file_read),
                black_box(&mut chunk_buffers),
            ));
        })
    });
}

fn bench_resolve_uniformity_from_load(c: &mut Criterion) {
    let chunk_coord = find_chunk_with_surface();
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = get_fbm();
    let uniformity = generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
    let column_range_map = ColumnRangeMap::new();
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
    c.bench_function("resolve_uniformity_from_load", |b| {
        b.iter(|| {
            black_box(resolve_uniformity(
                black_box(chunk_coord),
                black_box(&column_range_map),
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
    bench_resolve_uniformity_from_generation,
    bench_resolve_uniformity_from_load,
);
criterion_main!(benches);

//cargo bench --bench driver_bench -- build_full_mesh_and_spawn_with_collider
