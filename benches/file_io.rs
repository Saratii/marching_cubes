use std::{collections::HashSet, fs::create_dir_all, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use heed::{Database, EnvOpenOptions, types::SerdeBincode};
use marching_cubes::{
    data_loader::file_loader::{deserialize_chunk_data, serialize_chunk_data},
    terrain::{chunk_generator::generate_densities, terrain::TerrainChunk},
};

fn benchmark_read_single_chunk(c: &mut Criterion) {
    create_dir_all("data/test_db").unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024) // 10 GB max size
            .max_dbs(1)
            .open("data/test_db")
            .unwrap()
    };
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    c.bench_function("read_single_chunk", |b| {
        b.iter(|| {
            let rtxn = env.read_txn().unwrap();
            let bytes = db.get(&rtxn, &(0, 0, 0)).unwrap().unwrap();
            black_box(deserialize_chunk_data(&bytes));
        })
    });
}

fn benchmark_write_single_chunk(c: &mut Criterion) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let chunk = generate_densities(&(0, 0, 0), &fbm);
    let chunk = TerrainChunk { sdfs: chunk };
    let chunk_coord = (0, 0, 0);
    create_dir_all("data/test_db").unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024) // 10 GB max size
            .max_dbs(1)
            .open("data/test_db")
            .unwrap()
    };
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    let bytes = serialize_chunk_data(&chunk);
    c.bench_function("write_single_new_chunk", |b| {
        b.iter(|| {
            let mut wtxn = env.write_txn().unwrap();
            db.put(&mut wtxn, &chunk_coord, &bytes).unwrap();
            wtxn.commit().unwrap();
        })
    });
}

fn benchmark_bulk_read_chunks(c: &mut Criterion) {
    create_dir_all("data/test_db").unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024) // 10 GB max size
            .max_dbs(1)
            .open("data/test_db")
            .unwrap()
    };
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    let keys = {
        let rtxn = env.read_txn().unwrap();
        let keys: HashSet<(i16, i16, i16)> = db
            .iter(&rtxn)
            .unwrap()
            .take(100)
            .map(|result| result.unwrap().0)
            .collect();
        keys
    };
    c.bench_function("bulk_read_chunks", |b| {
        b.iter(|| {
            let rtxn = env.read_txn().unwrap();
            let mut chunks = Vec::new();
            for key in keys.iter() {
                let bytes = db.get(&rtxn, key).unwrap().unwrap();
                let chunk = deserialize_chunk_data(&bytes);
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
        let chunk = generate_densities(&chunk_coord, &noise_function);
        let terrain_chunk = TerrainChunk { sdfs: chunk };
        chunks_data.push((terrain_chunk, chunk_coord));
    }
    create_dir_all("data/test_db").unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024) // 10 GB max size
            .max_dbs(1)
            .open("data/test_db")
            .unwrap()
    };
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    c.bench_function("bulk_write_new_chunks", |b| {
        b.iter(|| {
            let mut wtxn = env.write_txn().unwrap();
            for (terrain_chunk, chunk_coord) in &chunks_data {
                let bytes = serialize_chunk_data(terrain_chunk);
                db.put(&mut wtxn, chunk_coord, &bytes).unwrap();
            }
            wtxn.commit().unwrap();
        })
    });
}

criterion_group!(
    benches,
    benchmark_bulk_read_chunks,
    benchmark_bulk_write_new_chunk,
    benchmark_write_single_chunk,
    benchmark_read_single_chunk,
);
criterion_main!(benches);

//cargo bench --bench file_io -- benchmark_read_single_chunk
