use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::{
    data_loader::chunk_loader::{
        create_chunk_file_data, load_chunk_data, load_chunk_index_map, update_chunk_file_data,
    },
    terrain::{chunk_generator::generate_densities, terrain::TerrainChunk},
};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::hint::black_box;

fn benchmark_generate_densities(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    c.bench_function("generate_densities_single_chunk", |b| {
        b.iter(|| generate_densities(black_box(&(0, 0, 0)), black_box(&noise_function)))
    });
}

fn benchmark_load_chunk_data(c: &mut Criterion) {
    let index_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let mut data_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let index_map = load_chunk_index_map(&index_file);
    let first_chunk_coord = *index_map.keys().next().unwrap();
    c.bench_function("load_chunk_data_single_chunk", |b| {
        b.iter(|| {
            let chunk = load_chunk_data(
                black_box(&mut data_file),
                black_box(&index_map),
                black_box(first_chunk_coord),
            );
            black_box(chunk);
        })
    });
}

fn benchmark_update_chunk_data(c: &mut Criterion) {
    let index_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let mut data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let index_map = load_chunk_index_map(&index_file);
    let first_chunk_coord = *index_map.keys().next().unwrap();
    let chunk = load_chunk_data(&mut data_file, &index_map, first_chunk_coord);
    c.bench_function("update_chunk_file_data_single_chunk", |b| {
        b.iter(|| {
            update_chunk_file_data(
                black_box(&index_map),
                black_box(first_chunk_coord),
                black_box(&chunk),
                black_box(&data_file),
            );
        })
    });
}

fn benchmark_create_chunk_file_data(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let chunk = generate_densities(&(0, 0, 0), &noise_function);
    let terrain_chunk = TerrainChunk { densities: chunk };
    let chunk_coord = (1000, 1000, 1000);
    let mut index_map = HashMap::new();
    let data_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("data/benchmark_chunk_data.txt")
        .unwrap();
    let index_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("data/benchmark_chunk_index_data.txt")
        .unwrap();
    c.bench_function("create_chunk_file_data", |b| {
        b.iter(|| {
            create_chunk_file_data(
                black_box(&terrain_chunk),
                black_box(chunk_coord),
                black_box(&mut index_map),
                black_box(&data_file),
                black_box(&index_file),
            );
        })
    });
}

criterion_group!(
    benches,
    benchmark_generate_densities,
    benchmark_load_chunk_data,
    benchmark_update_chunk_data,
    benchmark_create_chunk_file_data
);
criterion_main!(benches);
