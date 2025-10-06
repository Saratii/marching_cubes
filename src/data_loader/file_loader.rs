use crate::conversions::chunk_coord_to_world_pos;
use crate::player::player::{MainCameraTag, PlayerTag};
use crate::sparse_voxel_octree::ChunkSvo;
use crate::terrain::terrain::{
    CHUNK_SIZE, HALF_CHUNK, TerrainChunk, VOXELS_PER_CHUNK, VoxelData, Z1_RADIUS, Z2_RADIUS_SQUARED,
};
use bevy::prelude::*;
use bevy::render::primitives::{Aabb, Frustum};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

const BYTES_PER_VOXEL: usize = std::mem::size_of::<f32>() + std::mem::size_of::<u8>();
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
// - Number of voxels: u32 (4 bytes)
// - SDF values: num_voxels * f32 (4 bytes each)
// - Material values: num_voxels * u8 (1 byte each)

fn serialize_chunk_data(chunk: &TerrainChunk) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(CHUNK_SERIALIZED_SIZE);
    for voxel in chunk.sdfs.iter() {
        buffer.extend_from_slice(&voxel.sdf.to_le_bytes());
    }
    for voxel in chunk.sdfs.iter() {
        buffer.push(voxel.material);
    }
    buffer
}

pub fn deserialize_chunk_data(data: &[u8]) -> TerrainChunk {
    let mut sdfs = Vec::with_capacity(VOXELS_PER_CHUNK);
    let (sdf_bytes, material_bytes) = data.split_at(VOXELS_PER_CHUNK * 4);
    for (sdf_bytes, &material) in sdf_bytes.chunks_exact(4).zip(material_bytes) {
        let sdf = f32::from_le_bytes(sdf_bytes.try_into().unwrap());
        sdfs.push(VoxelData { sdf, material });
    }
    TerrainChunk {
        sdfs: sdfs.into_boxed_slice(),
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
    let total_size = VOXELS_PER_CHUNK * 4 + VOXELS_PER_CHUNK; //sdfs + materials
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
}

//this could be optimized by not calling it every frame
//loop through every loaded chunk and validate it against Z1 and Z2
pub fn try_deallocate(
    mut svo: ResMut<ChunkSvo>,
    mut commands: Commands,
    frustum: Single<&Frustum, With<MainCameraTag>>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let player_pos = player_transform.translation;
    let min_z1_cube = player_pos - Vec3::splat(Z1_RADIUS);
    let max_z1_cube = player_pos + Vec3::splat(Z1_RADIUS);

    // Collect all leaves to remove
    let mut leaves_to_remove = Vec::new();

    for leaf in svo.root.iter() {
        let leaf_world_pos = chunk_coord_to_world_pos(&leaf.position);
        let leaf_max = leaf_world_pos + Vec3::splat(leaf.size as f32 * CHUNK_SIZE);

        // Skip leaves inside Z1 cube
        if (leaf_max.x >= min_z1_cube.x && leaf_world_pos.x <= max_z1_cube.x)
            && (leaf_max.y >= min_z1_cube.y && leaf_world_pos.y <= max_z1_cube.y)
            && (leaf_max.z >= min_z1_cube.z && leaf_world_pos.z <= max_z1_cube.z)
        {
            continue;
        }

        // Check chunks for Z2 and frustum; if none are kept, mark leaf for removal
        let mut keep_any_chunk = false;
        for (entity, _chunk) in &leaf.chunks {
            let chunk_world_pos = chunk_coord_to_world_pos(&leaf.position);
            let distance_sq = chunk_world_pos.distance_squared(player_pos);
            let aabb = Aabb {
                center: chunk_world_pos.into(),
                half_extents: Vec3A::splat(HALF_CHUNK),
            };
            if distance_sq <= Z2_RADIUS_SQUARED && frustum.intersects_obb_identity(&aabb) {
                keep_any_chunk = true;
            } else {
                commands.entity(*entity).despawn();
            }
        }

        if !keep_any_chunk {
            leaves_to_remove.push(leaf.position);
        }
    }

    // Remove empty leaves from the SVO
    for pos in leaves_to_remove {
        svo.root.remove_leaf(pos);
    }
}
