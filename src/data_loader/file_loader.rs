use bevy::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::transmute;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::constants::SAMPLES_PER_CHUNK;
use crate::data_loader::column_range_map::ColumnRangeMap;
use crate::terrain::chunk_generator::MaterialCode;
use crate::terrain::terrain::Uniformity;

const BYTES_PER_VOXEL: usize = std::mem::size_of::<i16>() + std::mem::size_of::<u8>();
pub const CHUNK_SERIALIZED_SIZE: usize = SAMPLES_PER_CHUNK * BYTES_PER_VOXEL;
const TOMBSTONE_BYTES: [u8; 6] = [0xFF; 6];

#[derive(Resource)]
pub struct PlayerDataFile(pub File);

//when a non-uniform chunk becomes uniform and is removed from the main data file, mark its spot as available to be reused
#[derive(Resource)]
pub struct StaleCompressionFile(pub Arc<Mutex<File>>);

#[derive(Resource)]
pub struct ChunkEntityMap(pub FxHashMap<(i16, i16, i16), Entity>);

// Binary format layout:
// - SDF values: num_voxels * i16 (2 bytes each)
// - Material values: num_voxels * u8 (1 byte each)

//serialize densities and materials into a byte buffer
fn serialize_chunk_data(densities: &[i16], materials: &[MaterialCode], mut buffer: &mut [u8]) {
    for &d in densities.iter() {
        let (dst, rest) = buffer.split_at_mut(2);
        dst.copy_from_slice(&d.to_le_bytes());
        buffer = rest;
    }
    for &m in materials.iter() {
        buffer[0] = unsafe { transmute::<MaterialCode, u8>(m) };
        buffer = &mut buffer[1..];
    }
}

//read density and material data into provided buffers
pub fn deserialize_chunk_data(
    data: &[u8],
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
) {
    let (sdf_bytes, material_bytes) = data.split_at(SAMPLES_PER_CHUNK * 2);
    for (index, (sdf_chunk, &material)) in sdf_bytes.chunks_exact(2).zip(material_bytes).enumerate()
    {
        let density = i16::from_le_bytes([sdf_chunk[0], sdf_chunk[1]]);
        density_buffer[index] = density;
        material_buffer[index] = unsafe { transmute::<u8, MaterialCode>(material) };
    }
}

pub fn write_chunk(
    densities: &[i16],
    materials: &[MaterialCode],
    chunk_coord: &(i16, i16, i16),
    index_map_delta: &mut FxHashMap<(i16, i16, i16), u64>,
    chunk_data_file: &mut File,
    chunk_index_file: &mut File,
    index_buffer_allocation: &mut Vec<u8>,
    serial_buffer: &mut [u8],
) {
    index_buffer_allocation.clear();
    let byte_offset = chunk_data_file.seek(SeekFrom::End(0)).unwrap();
    serialize_chunk_data(densities, materials, serial_buffer);
    chunk_data_file.write_all(serial_buffer).unwrap();
    chunk_data_file.flush().unwrap();
    chunk_index_file.seek(SeekFrom::End(0)).unwrap();
    index_buffer_allocation.extend_from_slice(&chunk_coord.0.to_le_bytes());
    index_buffer_allocation.extend_from_slice(&chunk_coord.1.to_le_bytes());
    index_buffer_allocation.extend_from_slice(&chunk_coord.2.to_le_bytes());
    index_buffer_allocation.extend_from_slice(&byte_offset.to_le_bytes());
    chunk_index_file
        .write_all(&index_buffer_allocation)
        .unwrap();
    chunk_index_file.flush().unwrap();
    index_map_delta.insert(*chunk_coord, byte_offset);
}

pub fn update_chunk(
    byte_offset: u64,
    densities: &[i16],
    materials: &[MaterialCode],
    chunk_data_file: &mut File,
    serial_buffer: &mut [u8],
) {
    serialize_chunk_data(densities, materials, serial_buffer);
    chunk_data_file.seek(SeekFrom::Start(byte_offset)).unwrap();
    chunk_data_file.write_all(serial_buffer).unwrap();
    chunk_data_file.flush().unwrap();
}

//loads chunk data into provided density and material buffers
pub fn load_chunk(
    chunk_data_file: &mut File,
    byte_offset: u64,
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
) {
    chunk_data_file.seek(SeekFrom::Start(byte_offset)).unwrap();
    let mut buffer = [0u8; CHUNK_SERIALIZED_SIZE];
    chunk_data_file.read_exact(&mut buffer).unwrap();
    deserialize_chunk_data(&buffer, density_buffer, material_buffer);
}

pub fn load_chunk_index_map(index_file: &mut File) -> FxHashMap<(i16, i16, i16), u64> {
    let mut index_map = FxHashMap::default();
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
    create_dir_all(root.join("data/latest")).expect("Failed to create data directory");
    commands.insert_resource(ChunkEntityMap {
        0: FxHashMap::default(),
    }); //store entities on the main thread
    let player_data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/player_data.txt"))
        .unwrap();
    commands.insert_resource(PlayerDataFile(player_data_file));
    #[allow(unused)]
    let stale_chunks_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/stale_chunks.txt"))
        .unwrap();
}

// Load all chunk coords and track empty slots
pub fn load_uniform_chunks(
    f: &mut File,
    uniformity: Uniformity,
    column_range_map: &mut ColumnRangeMap,
) -> VecDeque<u64> {
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut data = Vec::new();
    f.read_to_end(&mut data).unwrap();
    // let count = data.len() / 6;
    let mut free_slots = VecDeque::new();
    for (i, b) in data.chunks_exact(6).enumerate() {
        let offset = (i * 6) as u64;
        if b == TOMBSTONE_BYTES {
            free_slots.push_back(offset);
        } else {
            let coord = (
                i16::from_le_bytes([b[0], b[1]]),
                i16::from_le_bytes([b[2], b[3]]),
                i16::from_le_bytes([b[4], b[5]]),
            );
            column_range_map.insert(coord, uniformity);
        }
    }
    free_slots
}

// Write either into a free slot or append
pub fn write_uniform_chunk(
    chunk_coord: &(i16, i16, i16),
    f: &mut File,
    free_slots: &mut VecDeque<u64>,
) {
    let mut buffer = [0; 6];
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

// Mark a chunk as deleted by overwriting with tombstone bytes
pub fn remove_uniform_chunk(
    chunk_coord: &(i16, i16, i16),
    f: &mut File,
    free_uniform_slots: &mut VecDeque<u64>,
) {
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = [0; 6];
    let mut offset = 0;
    while let Ok(_) = f.read_exact(&mut buffer) {
        if buffer != TOMBSTONE_BYTES {
            let x = i16::from_le_bytes([buffer[0], buffer[1]]);
            let y = i16::from_le_bytes([buffer[2], buffer[3]]);
            let z = i16::from_le_bytes([buffer[4], buffer[5]]);
            if (x, y, z) == *chunk_coord {
                f.seek(SeekFrom::Start(offset)).unwrap();
                f.write_all(&TOMBSTONE_BYTES).unwrap();
                free_uniform_slots.push_back(offset);
                break;
            }
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
