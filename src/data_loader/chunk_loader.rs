use crate::conversions::chunk_coord_to_world_pos;
use crate::player::player::{MainCameraTag, PlayerTag};
use crate::terrain::terrain::{
    CHUNK_SIZE, ChunkMap, L1_RADIUS_SQUARED, L2_RADIUS_SQUARED, TerrainChunk, VoxelData,
};
use bevy::math::Affine3A;
use bevy::prelude::*;
use bevy::render::primitives::{Aabb, Frustum};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

#[derive(Resource)]
pub struct ChunkIndexFile(pub File);

#[derive(Resource)]
pub struct ChunkDataFile(pub Arc<File>);

#[derive(Resource)]
pub struct ChunkIndexMap(pub Arc<Mutex<HashMap<(i16, i16, i16), u64>>>);

// Binary format layout:
// - Number of voxels: u32 (4 bytes)
// - SDF values: num_voxels * f32 (4 bytes each)
// - Material values: num_voxels * u8 (1 byte each)

fn serialize_chunk_data(chunk: &TerrainChunk) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(256);
    buffer.extend_from_slice(&(chunk.densities.len() as u32).to_le_bytes());
    for voxel in chunk.densities.iter() {
        buffer.extend_from_slice(&voxel.sdf.to_le_bytes());
    }
    for voxel in chunk.densities.iter() {
        buffer.push(voxel.material);
    }
    buffer
}

fn deserialize_chunk_data(data: &[u8]) -> TerrainChunk {
    let mut offset = 0;
    let num_voxels = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;
    let mut sdf_values = Vec::with_capacity(num_voxels);
    for _ in 0..num_voxels {
        let sdf = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        sdf_values.push(sdf);
        offset += 4;
    }
    let mut material_values = Vec::with_capacity(num_voxels);
    for _ in 0..num_voxels {
        material_values.push(data[offset]);
        offset += 1;
    }
    let densities: Box<[VoxelData]> = sdf_values
        .into_iter()
        .zip(material_values.into_iter())
        .map(|(sdf, material)| VoxelData { sdf, material })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let chunk = TerrainChunk { densities };
    chunk
}

pub fn create_chunk_file_data(
    chunk: &TerrainChunk,
    chunk_coord: (i16, i16, i16),
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
    index_map.insert(chunk_coord, byte_offset);
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

pub fn load_chunk_data(
    data_file: &File,
    index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_coord: &(i16, i16, i16),
) -> TerrainChunk {
    let byte_offset = *index_map.get(chunk_coord).unwrap();
    { data_file }.seek(SeekFrom::Start(byte_offset)).unwrap();
    let mut header = [0u8; 4];
    { data_file }.read_exact(&mut header).unwrap();
    let num_voxels = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let total_size = 4 + (num_voxels * 4) + num_voxels; // header + sdfs + materials
    { data_file }.seek(SeekFrom::Start(byte_offset)).unwrap();
    let mut buffer = vec![0u8; total_size];
    { data_file }.read_exact(&mut buffer).unwrap();
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
    println!("loaded {:?} chunk indexes from file", index_map.len());
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
    commands.insert_resource(ChunkDataFile(Arc::new(data_file)));
    commands.insert_resource(ChunkIndexMap(Arc::new(Mutex::new(load_chunk_index_map(
        &index_file,
    )))));
    commands.insert_resource(ChunkIndexFile(index_file));
}

//this could be optimized by not calling it every frame
//loop through every loaded chunk and validate it against L1 and L2
pub fn try_deallocate(
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    frustum: Single<&Frustum, With<MainCameraTag>>,
    player_position: Single<&Transform, With<PlayerTag>>,
) {
    #[cfg(feature = "timers")]
    let s = std::time::Instant::now();
    let mut aabb = Aabb {
        center: Vec3::ZERO.into(),
        half_extents: Vec3::splat(CHUNK_SIZE).into(),
    };
    chunk_map.0.retain(|chunk_coord, (entity, _chunk)| {
        let world_pos = chunk_coord_to_world_pos(chunk_coord);
        aabb.center = world_pos.into();
        let distance_squared = world_pos.distance_squared(player_position.translation);
        if distance_squared <= L1_RADIUS_SQUARED
            || frustum.intersects_obb(&aabb, &Affine3A::IDENTITY, true, true)
                && distance_squared < L2_RADIUS_SQUARED
        {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });
    #[cfg(feature = "timers")]
    {
        let duration = s.elapsed();
        println!("spent {:?} in deallocate_chunks", duration);
    }
}
