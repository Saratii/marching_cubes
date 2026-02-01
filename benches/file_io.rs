use criterion::{Criterion, criterion_group, criterion_main};
use marching_cubes::{
    data_loader::file_loader::{
        CHUNK_SERIALIZED_SIZE, load_chunk, load_chunk_index_map, update_chunk, write_chunk,
    },
    terrain::{
        chunk_generator::{calculate_chunk_start, get_fbm},
        terrain::{
            HEIGHTMAP_GRID_SIZE, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM,
            generate_chunk_into_buffers,
        },
    },
};
use rustc_hash::FxHashMap;
use std::{fs::OpenOptions, hint::black_box};

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
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    c.bench_function("read_single_chunk", |b| {
        b.iter(|| {
            black_box(load_chunk(
                black_box(&mut data_file),
                black_box(*offset),
                black_box(&mut density_buffer),
                black_box(&mut material_buffer),
            ));
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
    let byte_offset = index_map.get(&first_chunk_coord).unwrap();
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    load_chunk(
        &mut data_file,
        *byte_offset,
        &mut density_buffer,
        &mut material_buffer,
    );
    let mut serial_buffer = [0; CHUNK_SERIALIZED_SIZE];
    c.bench_function("write_single_existing_chunk", |b| {
        b.iter(|| {
            update_chunk(
                black_box(*byte_offset),
                black_box(&density_buffer),
                black_box(&material_buffer),
                black_box(&mut data_file),
                black_box(&mut serial_buffer),
            );
        })
    });
}

fn benchmark_write_single_new_chunk(c: &mut Criterion) {
    let fbm = get_fbm();
    let chunk_start = calculate_chunk_start(&(0, 0, 0));
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHTMAP_GRID_SIZE];
    generate_chunk_into_buffers(
        &fbm,
        chunk_start,
        &mut density_buffer,
        &mut material_buffer,
        &mut heightmap_buffer,
        SAMPLES_PER_CHUNK_DIM,
    );
    let chunk_coord = (1000, 1000, 1000);
    let mut index_map = FxHashMap::default();
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
    let mut buffer_reuse = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    let mut serial_buffer = [0; CHUNK_SERIALIZED_SIZE];
    c.bench_function("write_single_new_chunk", |b| {
        b.iter(|| {
            write_chunk(
                black_box(&density_buffer),
                black_box(&material_buffer),
                black_box(&chunk_coord),
                black_box(&mut index_map),
                black_box(&mut data_file),
                black_box(&mut index_file),
                black_box(&mut buffer_reuse),
                black_box(&mut serial_buffer),
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
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    c.bench_function("bulk_read_chunks", |b| {
        b.iter(|| {
            for &offset in &offsets {
                black_box(load_chunk(
                    black_box(&mut data_file),
                    black_box(offset),
                    black_box(&mut density_buffer),
                    black_box(&mut material_buffer),
                ));
            }
        })
    });
}

fn benchmark_bulk_write_new_chunk(c: &mut Criterion) {
    let fbm = get_fbm();
    let mut chunks_data = Vec::new();
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; HEIGHTMAP_GRID_SIZE];
    for i in 0..100 {
        let chunk_coord = (1000 + i, 1000, 1000);
        let chunk_start = calculate_chunk_start(&chunk_coord);
        generate_chunk_into_buffers(
            &fbm,
            chunk_start,
            &mut density_buffer,
            &mut material_buffer,
            &mut heightmap_buffer,
            SAMPLES_PER_CHUNK,
        );
        chunks_data.push((
            density_buffer.clone(),
            material_buffer.clone(),
            chunk_coord.clone(),
        ));
    }
    let mut buffer_reuse = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    let mut serial_buffer = [0; CHUNK_SERIALIZED_SIZE];
    c.bench_function("bulk_write_new_chunks", |b| {
        b.iter(|| {
            let mut index_map = FxHashMap::default();
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
            for (densities, materials, chunk_coord) in &chunks_data {
                write_chunk(
                    black_box(densities),
                    black_box(materials),
                    black_box(chunk_coord),
                    black_box(&mut index_map),
                    black_box(&mut data_file),
                    black_box(&mut index_file),
                    black_box(&mut buffer_reuse),
                    black_box(&mut serial_buffer),
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
