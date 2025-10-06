use std::sync::{Arc, atomic::AtomicBool};

use bevy::{
    ecs::{
        query::{Changed, With, Without},
        system::{Res, ResMut, Single},
    },
    math::Vec3,
    transform::components::Transform,
};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::driver::{ChunkChannels, ChunkRequest, ChunksBeingLoaded},
    player::player::{MainCameraTag, PlayerTag},
    sparse_voxel_octree::ChunkSvo,
    terrain::terrain::{Z1_RADIUS, Z2_RADIUS, Z2_RADIUS_SQUARED},
};

#[inline]
pub fn in_zone_1(player_position: &Vec3, chunk_center: &Vec3) -> bool {
    (player_position.x >= chunk_center.x - Z1_RADIUS)
        && (player_position.x <= chunk_center.x + Z1_RADIUS)
        && (player_position.y >= chunk_center.y - Z1_RADIUS)
        && (player_position.y <= chunk_center.y + Z1_RADIUS)
        && (player_position.z >= chunk_center.z - Z1_RADIUS)
        && (player_position.z <= chunk_center.z + Z1_RADIUS)
}

pub fn z1_chunk_load(
    player_transform: Single<&Transform, (With<PlayerTag>, Changed<Transform>)>,
    svo: Res<ChunkSvo>,
    chunk_channels: ResMut<ChunkChannels>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
) {
    #[cfg(feature = "timers")]
    let s = std::time::Instant::now();
    let min_world_pos = &player_transform.translation - Vec3::splat(Z1_RADIUS);
    let max_world_pos = &player_transform.translation + Vec3::splat(Z1_RADIUS);
    let min_chunk = world_pos_to_chunk_coord(&min_world_pos);
    let max_chunk = world_pos_to_chunk_coord(&max_world_pos);
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                if !svo.root.contains(&chunk_coord)
                    && !chunks_being_loaded.0.contains_key(&chunk_coord)
                {
                    let canceled_pointer = Arc::new(AtomicBool::new(false));
                    chunks_being_loaded
                        .0
                        .insert(chunk_coord, Arc::clone(&canceled_pointer));
                    chunk_channels
                        .requests
                        .send(ChunkRequest {
                            position: chunk_coord,
                            level: 0,
                            canceled: canceled_pointer,
                        })
                        .unwrap();
                }
            }
        }
    }
    #[cfg(feature = "timers")]
    {
        let duration = s.elapsed();
        if duration > std::time::Duration::from_micros(200) {
            println!("{:<40} {:?}", "z1_chunk_load", duration);
        }
    }
}

//load chunks within Z2 range.
pub fn z2_chunk_load(
    svo: Res<ChunkSvo>,
    player_transform: Single<&mut Transform, (With<PlayerTag>, Without<MainCameraTag>)>,
    chunk_channels: ResMut<ChunkChannels>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
) {
    #[cfg(feature = "timers")]
    let start = std::time::Instant::now();
    let player_pos = player_transform.translation;
    let min_world_pos = player_pos - Vec3::splat(Z2_RADIUS);
    let max_world_pos = player_pos + Vec3::splat(Z2_RADIUS);
    let min_chunk = world_pos_to_chunk_coord(&min_world_pos);
    let max_chunk = world_pos_to_chunk_coord(&max_world_pos);
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                let distance_squared = chunk_world_pos.distance_squared(player_pos);
                if distance_squared <= Z2_RADIUS_SQUARED {
                    if !svo.root.contains(&chunk_coord)
                        && !chunks_being_loaded.0.contains_key(&chunk_coord)
                    {
                        let canceled_pointer = Arc::new(AtomicBool::new(false));
                        chunks_being_loaded
                            .0
                            .insert(chunk_coord, Arc::clone(&canceled_pointer));
                        chunk_channels
                            .requests
                            .send(ChunkRequest {
                                position: chunk_coord,
                                level: 0,
                                canceled: canceled_pointer,
                            })
                            .unwrap();
                    }
                }
            }
        }
    }
    #[cfg(feature = "timers")]
    {
        let duration = start.elapsed();
        if duration > std::time::Duration::from_micros(200) {
            println!("{:<40} {:?}", "z2_chunk_load", duration);
        }
    }
}
