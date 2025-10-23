use crate::terrain::terrain::{NoiseFunction, TerrainChunk, VOXELS_PER_CHUNK};
use bevy::prelude::*;
use fastnoise2::SafeNode;
use fastnoise2::generator::simplex::opensimplex2;
use fastnoise2::generator::{Generator, GeneratorWrapper};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

const BYTES_PER_VOXEL: usize = std::mem::size_of::<i16>() + std::mem::size_of::<u8>();
const CHUNK_SERIALIZED_SIZE: usize = VOXELS_PER_CHUNK * BYTES_PER_VOXEL;

#[derive(Resource)]
pub struct ChunkIndexFile(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkDataFileRead(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkDataFileReadWrite(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkIndexMap(pub Arc<Mutex<HashMap<(i16, i16, i16), u64>>>);

// Binary format layout:
// - SDF values: num_voxels * i16 (2 bytes each)
// - Material values: num_voxels * u8 (1 byte each)

fn serialize_chunk_data(chunk: &TerrainChunk) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(CHUNK_SERIALIZED_SIZE);
    for density in chunk.densities.iter() {
        buffer.extend_from_slice(&density.to_le_bytes());
    }
    for material in chunk.materials.iter() {
        buffer.push(*material);
    }
    buffer
}

pub fn deserialize_chunk_data(data: &[u8]) -> TerrainChunk {
    let mut densities = Vec::with_capacity(VOXELS_PER_CHUNK);
    let mut materials = Vec::with_capacity(VOXELS_PER_CHUNK);
    let (sdf_bytes, material_bytes) = data.split_at(VOXELS_PER_CHUNK * 2);
    for (sdf_chunk, &material) in sdf_bytes.chunks_exact(2).zip(material_bytes) {
        let density = i16::from_le_bytes([sdf_chunk[0], sdf_chunk[1]]);
        densities.push(density);
        materials.push(material);
    }
    TerrainChunk {
        densities: densities.into_boxed_slice(),
        materials: materials.into_boxed_slice(),
    }
}

pub fn create_chunk_file_data(
    chunk: &TerrainChunk,
    chunk_coord: &(i16, i16, i16),
    index_map: &mut HashMap<(i16, i16, i16), u64>,
    mut data_file: &File,
    mut index_file: &File,
) {
    let byte_offset = data_file.seek(SeekFrom::End(0)).unwrap();
    let buffer = serialize_chunk_data(chunk);
    data_file.write_all(&buffer).unwrap();
    let mut index_buffer = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    index_buffer.extend_from_slice(&chunk_coord.0.to_le_bytes());
    index_buffer.extend_from_slice(&chunk_coord.1.to_le_bytes());
    index_buffer.extend_from_slice(&chunk_coord.2.to_le_bytes());
    index_buffer.extend_from_slice(&byte_offset.to_le_bytes());
    index_file.write_all(&index_buffer).unwrap();
    index_map.insert(*chunk_coord, byte_offset);
}

pub fn update_chunk_file_data(
    index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_coord: (i16, i16, i16),
    chunk: &TerrainChunk,
    mut data_file: &File,
) {
    let byte_offset = index_map.get(&chunk_coord).unwrap();
    let buffer = serialize_chunk_data(chunk);
    data_file.seek(SeekFrom::Start(*byte_offset)).unwrap();
    data_file.write_all(&buffer).unwrap();
}

pub fn load_chunk_data(data_file: &mut File, byte_offset: u64) -> TerrainChunk {
    let total_size = VOXELS_PER_CHUNK * 2 + VOXELS_PER_CHUNK; // i16 sdfs (2 bytes) + u8 materials (1 byte)
    data_file.seek(SeekFrom::Start(byte_offset)).unwrap();
    let mut buffer = vec![0u8; total_size];
    data_file.read_exact(&mut buffer).unwrap();
    deserialize_chunk_data(&buffer)
}

pub fn load_chunk_index_map(mut index_file: &File) -> HashMap<(i16, i16, i16), u64> {
    let mut index_map = HashMap::new();
    index_file.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = [0u8; 14]; // sizeof (i16, i16, i16, u64)
    while let Ok(_) = index_file.read_exact(&mut buffer) {
        let x = i16::from_le_bytes([buffer[0], buffer[1]]);
        let y = i16::from_le_bytes([buffer[2], buffer[3]]);
        let z = i16::from_le_bytes([buffer[4], buffer[5]]);
        let offset = u64::from_le_bytes([
            buffer[6], buffer[7], buffer[8], buffer[9], buffer[10], buffer[11], buffer[12],
            buffer[13],
        ]);
        index_map.insert((x, y, z), offset);
    }
    index_map
}

pub fn setup_chunk_loading(mut commands: Commands) {
    let index_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/chunk_data.txt")
        .unwrap();
    let data_file_read = OpenOptions::new()
        .read(true)
        .open("data/chunk_data.txt")
        .unwrap();
    commands.insert_resource(ChunkDataFileReadWrite(Arc::new(Mutex::new(data_file))));
    commands.insert_resource(ChunkDataFileRead(Arc::new(Mutex::new(data_file_read))));
    commands.insert_resource(ChunkIndexMap(Arc::new(Mutex::new(load_chunk_index_map(
        &index_file,
    )))));
    commands.insert_resource(ChunkIndexFile(Arc::new(Mutex::new(index_file))));
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    commands.insert_resource(NoiseFunction(Arc::new(fbm)));
}
