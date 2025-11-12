use std::{collections::HashMap, fs::OpenOptions, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::{
    data_loader::file_loader::{
        create_chunk_file_data, load_chunk_data, load_chunk_index_map, update_chunk_file_data,
    },
    terrain::{
        chunk_generator::generate_densities,
        terrain::{TerrainChunk, UniformChunk},
    },
};

fn benchmark_read_single_chunk(c: &mut Criterion) {
    let mut index_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let mut data_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let index_map = load_chunk_index_map(&mut index_file);
    let offset = index_map.get(&(0, 0, 0)).unwrap();
    c.bench_function("read_single_chunk", |b| {
        b.iter(|| {
            let chunk = load_chunk_data(black_box(&mut data_file), black_box(*offset));
            black_box(chunk);
        })
    });
}

fn benchmark_write_single_existing_chunk(c: &mut Criterion) {
    let mut index_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let mut data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let index_map = load_chunk_index_map(&mut index_file);
    let first_chunk_coord = *index_map.keys().next().unwrap();
    let file_offset = index_map.get(&first_chunk_coord).unwrap();
    let chunk = load_chunk_data(&mut data_file, *file_offset);
    c.bench_function("write_single_existing_chunk", |b| {
        b.iter(|| {
            update_chunk_file_data(
                black_box(&index_map),
                black_box(first_chunk_coord),
                black_box(&chunk),
                black_box(&mut data_file),
            );
        })
    });
}

fn benchmark_write_single_new_chunk(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let (densities, materials, _) = generate_densities(&(0, 0, 0), &noise_function);
    let terrain_chunk = TerrainChunk {
        densities,
        materials,
        is_uniform: UniformChunk::NonUniform,
    };
    let chunk_coord = (1000, 1000, 1000);
    let mut index_map = HashMap::new();
    let mut data_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("data/benchmark_chunk_data.txt")
        .unwrap();
    let mut index_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("data/benchmark_chunk_index_data.txt")
        .unwrap();
    c.bench_function("write_single_new_chunk", |b| {
        b.iter(|| {
            create_chunk_file_data(
                black_box(&terrain_chunk),
                black_box(&chunk_coord),
                black_box(&mut index_map),
                black_box(&mut data_file),
                black_box(&mut index_file),
            );
        })
    });
}

fn benchmark_bulk_read_chunks(c: &mut Criterion) {
    let mut index_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let mut data_file = OpenOptions::new()
        .read(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let index_map = load_chunk_index_map(&mut index_file);
    let chunk_coords: Vec<_> = index_map.keys().take(100).cloned().collect();
    let offsets: Vec<_> = chunk_coords
        .iter()
        .map(|coord| *index_map.get(coord).unwrap())
        .collect();
    c.bench_function("bulk_read_chunks", |b| {
        b.iter(|| {
            let mut chunks = Vec::new();
            for &offset in &offsets {
                let chunk = load_chunk_data(black_box(&mut data_file), black_box(offset));
                chunks.push(chunk);
            }
            black_box(chunks);
        })
    });
}

fn benchmark_bulk_write_new_chunk(c: &mut Criterion) {
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut chunks_data = Vec::new();
    for i in 0..100 {
        let chunk_coord = (1000 + i, 1000, 1000);
        let (densities, materials, _) = generate_densities(&chunk_coord, &noise_function);
        let terrain_chunk = TerrainChunk {
            densities,
            materials,
            is_uniform: UniformChunk::NonUniform,
        };
        chunks_data.push((terrain_chunk, chunk_coord));
    }
    c.bench_function("bulk_write_new_chunks", |b| {
        b.iter(|| {
            let mut index_map = HashMap::new();
            let mut data_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("data/benchmark_chunk_data.txt")
                .unwrap();
            let mut index_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("data/benchmark_chunk_index_data.txt")
                .unwrap();
            for (terrain_chunk, chunk_coord) in &chunks_data {
                create_chunk_file_data(
                    black_box(terrain_chunk),
                    black_box(chunk_coord),
                    black_box(&mut index_map),
                    black_box(&mut data_file),
                    black_box(&mut index_file),
                );
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_read_single_chunk,
    benchmark_write_single_existing_chunk,
    benchmark_write_single_new_chunk,
    benchmark_bulk_read_chunks,
    benchmark_bulk_write_new_chunk,
);
criterion_main!(benches);

//cargo bench --bench file_io -- benchmark_read_single_chunk
