use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::{
    data_loader::chunk_loader::{
        create_chunk_file_data, deserialize_chunk_data, load_chunk_data, load_chunk_index_map,
        update_chunk_file_data,
    },
    terrain::{
        chunk_generator::generate_densities,
        terrain::{L1_RADIUS_SQUARED, TerrainChunk, VOXELS_PER_CHUNK},
    },
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
    c.bench_function("load_chunk_data_single_chunk", |b| {
        b.iter(|| {
            let chunk = load_chunk_data(
                black_box(&mut data_file),
                black_box(&index_map),
                black_box(&(0, -5, 0)),
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
    let chunk = load_chunk_data(&mut data_file, &index_map, &first_chunk_coord);
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
    let terrain_chunk = TerrainChunk { sdfs: chunk };
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

fn bench_iterate_exterior(c: &mut Criterion) {
    const CHUNK_CREATION_RADIUS: f32 = 100.0;
    const CHUNK_SIZE: f32 = 3.1;
    c.bench_function("iterate_exterior", |b| {
        b.iter(|| {
            let max_voxel = (CHUNK_CREATION_RADIUS / CHUNK_SIZE as f32).ceil() as i32;
            let mut count = 0;
            for x in -max_voxel..=max_voxel {
                let dx = x as f32 * CHUNK_SIZE as f32;
                for y in -max_voxel..=max_voxel {
                    let dy = y as f32 * CHUNK_SIZE as f32;
                    let r_sq_xy = L1_RADIUS_SQUARED - dx * dx - dy * dy;
                    if r_sq_xy < 0.0 {
                        continue;
                    }
                    let max_z = (r_sq_xy.sqrt() / CHUNK_SIZE as f32).floor() as i32;
                    for z in -max_z..=max_z {
                        if is_exterior_chunk(x, y, z, L1_RADIUS_SQUARED, CHUNK_SIZE) {
                            count += 1;
                        }
                    }
                }
            }
            println!("Exterior chunks: {}", count);
        })
    });
}

fn benchmark_deserialize_chunk_data(c: &mut Criterion) {
    let buffer_size = VOXELS_PER_CHUNK * (4 + 1);
    let mut data = vec![0u8; buffer_size];
    for i in 0..VOXELS_PER_CHUNK {
        let sdf_bytes = (i as f32).to_le_bytes();
        data[i * 4..i * 4 + 4].copy_from_slice(&sdf_bytes);
        data[VOXELS_PER_CHUNK * 4 + i] = (i % 256) as u8;
    }
    c.bench_function("deserialize_chunk_data", |b| {
        b.iter(|| {
            let chunk = deserialize_chunk_data(black_box(&data));
            black_box(chunk);
        })
    });
}

criterion_group!(
    benches,
    benchmark_generate_densities,
    benchmark_load_chunk_data,
    benchmark_update_chunk_data,
    benchmark_create_chunk_file_data,
    bench_iterate_exterior,
    benchmark_deserialize_chunk_data,
);
criterion_main!(benches);

// Returns true if the chunk at (x, y, z) is on the exterior of the sphere
fn is_exterior_chunk(x: i32, y: i32, z: i32, radius_sq: f32, chunk_size: f32) -> bool {
    let dx = x as f32 * chunk_size as f32;
    let dy = y as f32 * chunk_size as f32;
    let dz = z as f32 * chunk_size as f32;
    let dist_sq = dx * dx + dy * dy + dz * dz;
    if dist_sq > radius_sq {
        return false;
    }
    for &nx in &[x - 1, x + 1] {
        let ndx = nx as f32 * chunk_size as f32;
        if ndx * ndx + dy * dy + dz * dz > radius_sq {
            return true;
        }
    }
    for &ny in &[y - 1, y + 1] {
        let ndy = ny as f32 * chunk_size as f32;
        if dx * dx + ndy * ndy + dz * dz > radius_sq {
            return true;
        }
    }
    for &nz in &[z - 1, z + 1] {
        let ndz = nz as f32 * chunk_size as f32;
        if dx * dx + dy * dy + ndz * ndz > radius_sq {
            return true;
        }
    }
    false
}

//cargo bench --bench chunk_generation -- deserialize_chunk_data
