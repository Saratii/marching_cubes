use crate::terrain::terrain::{SAMPLES_PER_CHUNK, TerrainChunk, UniformChunk};
use bevy::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const BYTES_PER_VOXEL: usize = std::mem::size_of::<i16>() + std::mem::size_of::<u8>();
const CHUNK_SERIALIZED_SIZE: usize = SAMPLES_PER_CHUNK * BYTES_PER_VOXEL;

#[derive(Resource)]
pub struct ChunkIndexFile(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkDataFileRead(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkDataFileReadWrite(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct PlayerDataFile(pub File);

#[derive(Resource)]
pub struct CompressionFileHandles {
    pub dirt_file: Arc<Mutex<File>>,
    pub air_file: Arc<Mutex<File>>,
}

//when a non-uniform chunk becomes uniform and is removed from the main data file, mark its spot as available to be reused
#[derive(Resource)]
pub struct StaleCompressionFile(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct UniformChunkMap {
    pub air_chunks: Arc<Mutex<HashSet<(i16, i16, i16)>>>,
    pub dirt_chunks: Arc<Mutex<HashSet<(i16, i16, i16)>>>,
    pub uniform_air_empty_offsets: Arc<Mutex<VecDeque<u64>>>,
    pub uniform_dirt_empty_offsets: Arc<Mutex<VecDeque<u64>>>,
}

#[derive(Resource)]
pub struct ChunkIndexMap(pub Arc<Mutex<HashMap<(i16, i16, i16), u64>>>);

#[derive(Resource)]
pub struct ChunkEntityMap(pub HashMap<(i16, i16, i16), Entity>);

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
    let mut densities = Vec::with_capacity(SAMPLES_PER_CHUNK);
    let mut materials = Vec::with_capacity(SAMPLES_PER_CHUNK);
    let (sdf_bytes, material_bytes) = data.split_at(SAMPLES_PER_CHUNK * 2);
    for (sdf_chunk, &material) in sdf_bytes.chunks_exact(2).zip(material_bytes) {
        let density = i16::from_le_bytes([sdf_chunk[0], sdf_chunk[1]]);
        densities.push(density);
        materials.push(material);
    }
    TerrainChunk {
        densities: densities.into_boxed_slice(),
        materials: materials.into_boxed_slice(),
        is_uniform: UniformChunk::NonUniform,
    }
}

pub fn create_chunk_file_data(
    chunk: &TerrainChunk,
    chunk_coord: &(i16, i16, i16),
    chunk_index_map: &mut HashMap<(i16, i16, i16), u64>,
    chunk_data_file: &mut File,
    chunk_index_file: &mut File,
) {
    let byte_offset = chunk_data_file.seek(SeekFrom::End(0)).unwrap();
    let buffer = serialize_chunk_data(chunk);
    chunk_data_file.write_all(&buffer).unwrap();
    chunk_data_file.flush().unwrap();
    chunk_index_file.seek(SeekFrom::End(0)).unwrap();
    let mut index_buffer = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    index_buffer.extend_from_slice(&chunk_coord.0.to_le_bytes());
    index_buffer.extend_from_slice(&chunk_coord.1.to_le_bytes());
    index_buffer.extend_from_slice(&chunk_coord.2.to_le_bytes());
    index_buffer.extend_from_slice(&byte_offset.to_le_bytes());
    chunk_index_file.write_all(&index_buffer).unwrap();
    chunk_index_file.flush().unwrap();
    chunk_index_map.insert(*chunk_coord, byte_offset);
}

pub fn update_chunk_file_data(
    chunk_index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_coord: (i16, i16, i16),
    chunk: &TerrainChunk,
    chunk_data_file: &mut File,
) {
    let byte_offset = chunk_index_map.get(&chunk_coord).unwrap();
    let buffer = serialize_chunk_data(chunk);
    chunk_data_file.seek(SeekFrom::Start(*byte_offset)).unwrap();
    chunk_data_file.write_all(&buffer).unwrap();
    chunk_data_file.flush().unwrap();
}

pub fn load_chunk_data(chunk_data_file: &mut File, byte_offset: u64) -> TerrainChunk {
    let total_size = SAMPLES_PER_CHUNK * 2 + SAMPLES_PER_CHUNK; // i16 sdfs (2 bytes) + u8 materials (1 byte)
    chunk_data_file.seek(SeekFrom::Start(byte_offset)).unwrap();
    let mut buffer = vec![0u8; total_size];
    chunk_data_file.read_exact(&mut buffer).unwrap();
    deserialize_chunk_data(&buffer)
}

pub fn load_chunk_index_map(index_file: &mut File) -> HashMap<(i16, i16, i16), u64> {
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

pub fn get_project_root() -> PathBuf {
    let exe_path = std::env::current_exe().expect("Failed to get executable path");
    exe_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("Failed to get project root")
        .to_path_buf()
}

pub fn setup_chunk_loading(mut commands: Commands) {
    let root = get_project_root();
    commands.insert_resource(ChunkEntityMap { 0: HashMap::new() }); //store entities on the main thread
    let player_data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/player_data.txt"))
        .unwrap();
    commands.insert_resource(PlayerDataFile(player_data_file));
    let mut index_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/chunk_index_data.txt"))
        .unwrap();
    let data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/chunk_data.txt"))
        .unwrap();
    let data_file_read = OpenOptions::new()
        .read(true)
        .open(root.join("data/chunk_data.txt"))
        .unwrap();
    let air_compression_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/air_compression_data.txt"))
        .unwrap();
    let dirt_compression_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/dirt_compression_data.txt"))
        .unwrap();
    #[allow(unused)]
    let stale_chunks_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/stale_chunks.txt"))
        .unwrap();
    commands.insert_resource(ChunkDataFileReadWrite(Arc::new(Mutex::new(data_file))));
    commands.insert_resource(ChunkDataFileRead(Arc::new(Mutex::new(data_file_read))));
    commands.insert_resource(CompressionFileHandles {
        dirt_file: Arc::new(Mutex::new(dirt_compression_file)),
        air_file: Arc::new(Mutex::new(air_compression_file)),
    });
    let index_map = load_chunk_index_map(&mut index_file);
    println!("Loaded {} chunks into index map", index_map.len());
    commands.insert_resource(ChunkIndexMap(Arc::new(Mutex::new(index_map))));
    commands.insert_resource(ChunkIndexFile(Arc::new(Mutex::new(index_file))));
}

// Load all chunk coords and track empty slots
pub fn load_uniform_chunks(f: &mut File) -> (HashSet<(i16, i16, i16)>, VecDeque<u64>) {
    let mut uniform_chunks = HashSet::new();
    let mut free_slots = VecDeque::new();
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = [0u8; 6];
    let mut offset = 0u64;
    while let Ok(_) = f.read_exact(&mut buffer) {
        let x = i16::from_le_bytes([buffer[0], buffer[1]]);
        let y = i16::from_le_bytes([buffer[2], buffer[3]]);
        let z = i16::from_le_bytes([buffer[4], buffer[5]]);
        if (x, y, z) == (0, 0, 0) {
            // treat (0,0,0) as a deleted slot
            free_slots.push_back(offset);
        } else {
            uniform_chunks.insert((x, y, z));
        }
        offset += 6;
    }
    (uniform_chunks, free_slots)
}

// Write either into a free slot or append
pub fn write_uniform_chunk(
    chunk_coord: &(i16, i16, i16),
    f: &mut File,
    free_slots: &mut VecDeque<u64>,
) {
    let mut buffer = [0u8; 6];
    buffer[..2].copy_from_slice(&chunk_coord.0.to_le_bytes());
    buffer[2..4].copy_from_slice(&chunk_coord.1.to_le_bytes());
    buffer[4..6].copy_from_slice(&chunk_coord.2.to_le_bytes());
    if let Some(pos) = free_slots.pop_front() {
        f.seek(SeekFrom::Start(pos)).unwrap();
    } else {
        f.seek(SeekFrom::End(0)).unwrap();
    }
    f.write_all(&buffer).unwrap();
    f.flush().unwrap();
}

// Mark a chunk as deleted by overwriting with zeros
pub fn remove_uniform_chunk(
    chunk_coord: &(i16, i16, i16),
    f: &mut File,
    free_uniform_slots: &mut VecDeque<u64>,
) {
    let target = chunk_coord;
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = [0u8; 6];
    let mut offset = 0u64;
    while let Ok(_) = f.read_exact(&mut buffer) {
        let x = i16::from_le_bytes([buffer[0], buffer[1]]);
        let y = i16::from_le_bytes([buffer[2], buffer[3]]);
        let z = i16::from_le_bytes([buffer[4], buffer[5]]);
        if (x, y, z) == *target {
            f.seek(SeekFrom::Start(offset)).unwrap();
            f.write_all(&[0; 6]).unwrap();
            free_uniform_slots.push_back(offset);
            break;
        }
        offset += 6;
    }
    f.flush().unwrap();
}

pub fn write_player_position(f: &mut File, pos: Vec3) {
    f.set_len(0).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    let s = format!("{} {} {}", pos.x, pos.y, pos.z);
    f.write_all(s.as_bytes()).unwrap();
    f.flush().unwrap();
}

pub fn read_player_position(f: &mut File) -> Option<Vec3> {
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut buf = String::new();
    if f.read_to_string(&mut buf).is_err() {
        return None;
    }
    let mut it = buf.split_whitespace();
    let x = it.next()?.parse::<f32>().ok()?;
    let y = it.next()?.parse::<f32>().ok()?;
    let z = it.next()?.parse::<f32>().ok()?;
    Some(Vec3::new(x, y, z))
}
