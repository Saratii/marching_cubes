use bevy::math::Vec3;

use crate::terrain::terrain::{CHUNK_SIZE, HALF_CHUNK};

pub fn chunk_coord_to_world_pos(chunk_coord: (i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE,
        chunk_coord.1 as f32 * CHUNK_SIZE,
        chunk_coord.2 as f32 * CHUNK_SIZE,
    )
}

pub fn world_pos_to_chunk_coord(world_pos: Vec3) -> (i16, i16, i16) {
    // Offset world position so (0,0,0) is at chunk center
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
