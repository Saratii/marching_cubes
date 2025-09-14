use bevy::math::Vec3;

use crate::terrain::terrain::{CHUNK_SIZE, HALF_CHUNK, VOXEL_SIZE};

pub fn chunk_coord_to_world_pos(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE,
        chunk_coord.1 as f32 * CHUNK_SIZE,
        chunk_coord.2 as f32 * CHUNK_SIZE,
    )
}

pub fn world_pos_to_chunk_coord(world_pos: &Vec3) -> (i16, i16, i16) {
    let offset_pos = Vec3 {
        x: world_pos.x + HALF_CHUNK,
        y: world_pos.y + HALF_CHUNK,
        z: world_pos.z + HALF_CHUNK,
    };
    let chunk_x = (offset_pos.x / CHUNK_SIZE).floor() as i16;
    let chunk_y = (offset_pos.y / CHUNK_SIZE).floor() as i16;
    let chunk_z = (offset_pos.z / CHUNK_SIZE).floor() as i16;
    (chunk_x, chunk_y, chunk_z)
}

pub fn world_pos_to_voxel_index(
    world_pos: &Vec3,
    chunk_coord: &(i16, i16, i16),
) -> (u32, u32, u32) {
    let chunk_world_center = chunk_coord_to_world_pos(&chunk_coord);
    let chunk_world_min = chunk_world_center - Vec3::splat(HALF_CHUNK);
    let relative_pos = world_pos - chunk_world_min;
    let voxel_x = (relative_pos.x / VOXEL_SIZE).floor() as u32;
    let voxel_y = (relative_pos.y / VOXEL_SIZE).floor() as u32;
    let voxel_z = (relative_pos.z / VOXEL_SIZE).floor() as u32;
    (voxel_x, voxel_y, voxel_z)
}

pub fn flatten_index(x: u32, y: u32, z: u32, dimension_size: usize) -> u32 {
    z * dimension_size as u32 * dimension_size as u32 + y * dimension_size as u32 + x
}
