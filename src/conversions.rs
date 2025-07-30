use bevy::math::Vec3;

use crate::terrain_generation::CHUNK_SIZE;

pub fn chunk_coord_to_world_pos(chunk_coord: (i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE,
        chunk_coord.1 as f32 * CHUNK_SIZE,
        chunk_coord.2 as f32 * CHUNK_SIZE,
    )
}
