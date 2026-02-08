use bevy::math::Vec3;

use crate::constants::{
    CHUNK_WORLD_SIZE, CHUNKS_PER_CLUSTER_DIM, CLUSTER_WORLD_LENGTH, HALF_CHUNK, VOXEL_WORLD_SIZE,
};

pub fn chunk_coord_to_world_pos(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_WORLD_SIZE,
        chunk_coord.1 as f32 * CHUNK_WORLD_SIZE,
        chunk_coord.2 as f32 * CHUNK_WORLD_SIZE,
    )
}

pub fn world_pos_to_chunk_coord(world_pos: &Vec3) -> (i16, i16, i16) {
    let offset_pos = Vec3 {
        x: world_pos.x + HALF_CHUNK,
        y: world_pos.y + HALF_CHUNK,
        z: world_pos.z + HALF_CHUNK,
    };
    let chunk_x = (offset_pos.x / CHUNK_WORLD_SIZE).floor() as i16;
    let chunk_y = (offset_pos.y / CHUNK_WORLD_SIZE).floor() as i16;
    let chunk_z = (offset_pos.z / CHUNK_WORLD_SIZE).floor() as i16;
    (chunk_x, chunk_y, chunk_z)
}

pub fn world_pos_to_voxel_index(
    world_pos: &Vec3,
    chunk_coord: &(i16, i16, i16),
) -> (u32, u32, u32) {
    let chunk_world_center = chunk_coord_to_world_pos(&chunk_coord);
    let chunk_world_min = chunk_world_center - Vec3::splat(HALF_CHUNK);
    let relative_pos = world_pos - chunk_world_min;
    let voxel_x = (relative_pos.x / VOXEL_WORLD_SIZE).floor() as u32;
    let voxel_y = (relative_pos.y / VOXEL_WORLD_SIZE).floor() as u32;
    let voxel_z = (relative_pos.z / VOXEL_WORLD_SIZE).floor() as u32;
    (voxel_x, voxel_y, voxel_z)
}

pub fn flatten_index(x: u32, y: u32, z: u32, dimension_size: usize) -> u32 {
    z * dimension_size as u32 * dimension_size as u32 + y * dimension_size as u32 + x
}

pub fn chunk_coord_to_cluster_coord(chunk_coord: &(i16, i16, i16)) -> (i16, i16, i16) {
    (
        chunk_coord.0.div_euclid(CHUNKS_PER_CLUSTER_DIM as i16),
        chunk_coord.1.div_euclid(CHUNKS_PER_CLUSTER_DIM as i16),
        chunk_coord.2.div_euclid(CHUNKS_PER_CLUSTER_DIM as i16),
    )
}

pub fn cluster_coord_to_world_pos(cluster_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        cluster_coord.0 as f32 * CLUSTER_WORLD_LENGTH,
        cluster_coord.1 as f32 * CLUSTER_WORLD_LENGTH,
        cluster_coord.2 as f32 * CLUSTER_WORLD_LENGTH,
    )
}

pub fn world_pos_to_cluster_coord(world_pos: &Vec3) -> (i16, i16, i16) {
    let half_cluster = CLUSTER_WORLD_LENGTH * 0.5;
    let offset_pos = Vec3 {
        x: world_pos.x + half_cluster,
        y: world_pos.y + half_cluster,
        z: world_pos.z + half_cluster,
    };
    let cluster_x = (offset_pos.x / CLUSTER_WORLD_LENGTH).floor() as i16;
    let cluster_y = (offset_pos.y / CLUSTER_WORLD_LENGTH).floor() as i16;
    let cluster_z = (offset_pos.z / CLUSTER_WORLD_LENGTH).floor() as i16;
    (cluster_x, cluster_y, cluster_z)
}

pub fn cluster_coord_to_min_chunk_coord(cluster_coord: &(i16, i16, i16)) -> (i16, i16, i16) {
    (
        cluster_coord.0 * CHUNKS_PER_CLUSTER_DIM as i16,
        cluster_coord.1 * CHUNKS_PER_CLUSTER_DIM as i16,
        cluster_coord.2 * CHUNKS_PER_CLUSTER_DIM as i16,
    )
}

#[inline(always)]
pub fn cluster_coord_to_world_center(c: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        c.0 as f32 * CLUSTER_WORLD_LENGTH + CLUSTER_WORLD_LENGTH * 0.5,
        c.1 as f32 * CLUSTER_WORLD_LENGTH + CLUSTER_WORLD_LENGTH * 0.5,
        c.2 as f32 * CLUSTER_WORLD_LENGTH + CLUSTER_WORLD_LENGTH * 0.5,
    )
}
