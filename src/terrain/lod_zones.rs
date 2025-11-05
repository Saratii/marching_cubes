//Zone 0: small circle around player with highest priority. Calculated outside of large traversal.

use bevy::{
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, ResMut, Single},
    },
    math::Vec3,
    transform::components::Transform,
};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::driver::{ChunkChannels, ChunkRequest, ChunksBeingLoaded},
    player::player::{MainCameraTag, PlayerTag},
    sparse_voxel_octree::ChunkSvo,
    terrain::terrain::{Z0_RADIUS, Z0_RADIUS_SQUARED},
};

pub const Z2_RADIUS: f32 = 500.0;
pub const Z2_RADIUS_SQUARED: f32 = Z2_RADIUS * Z2_RADIUS;

pub fn z2_chunk_load(
    mut svo: ResMut<ChunkSvo>,
    player_transform: Single<&mut Transform, (With<PlayerTag>, Without<MainCameraTag>)>,
    mut chunk_channels: ResMut<ChunkChannels>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    mut commands: Commands,
) {
    #[cfg(feature = "timers")]
    let start = std::time::Instant::now();
    let start = std::time::Instant::now();
    let mut chunks_to_deallocate: Vec<((i16, i16, i16), Option<Entity>)> = Vec::new();
    svo.root.query_chunks_outside_sphere(
        &player_transform.translation,
        Z2_RADIUS,
        &mut chunks_to_deallocate,
    );
    println!(
        "Time for query_chunks_outside_sphere: {:?}",
        start.elapsed()
    );
    let start = std::time::Instant::now();
    for (chunk_coord, entity) in &chunks_to_deallocate {
        chunks_being_loaded.0.remove(chunk_coord);
        svo.root.delete(*chunk_coord);
        if let Some(entity) = entity {
            commands.entity(*entity).despawn();
        }
    }
    println!("Time for deallocating chunks: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    svo.root.fill_missing_chunks_in_radius(
        &player_transform.translation,
        Z2_RADIUS,
        &mut chunks_being_loaded,
        &mut chunk_channels,
    );
    println!(
        "Time for fill_missing_chunks_in_radius: {:?}",
        start.elapsed()
    );
    #[cfg(feature = "timers")]
    {
        let duration = start.elapsed();
        if duration > std::time::Duration::from_micros(200) {
            println!("{:<40} {:?}", "z2_chunk_load", duration);
        }
    }
}

//For every chunk in the spherical bounding radius Z0_RADIUS around the player, every frame check if loaded
//If not loaded and not being loaded, create a atomic reference to track cancellation.
//Create a new request id by incrementing the last used id. Insert into the chunks being loaded map
//Send chunk load with priority 0 (highest)
//If chunk is loaded with priority 1, send a load request with priority 0 to upgrade true
pub fn z0_chunk_load(
    player_position: Single<&Transform, With<PlayerTag>>,
    chunk_svo: ResMut<ChunkSvo>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    chunk_channels: ResMut<ChunkChannels>,
) {
    let min_world_pos = player_position.translation - Vec3::splat(Z0_RADIUS);
    let max_world_pos = player_position.translation + Vec3::splat(Z0_RADIUS);
    let min_chunk = world_pos_to_chunk_coord(&min_world_pos);
    let max_chunk = world_pos_to_chunk_coord(&max_world_pos);
    for chunk in min_chunk.0..=max_chunk.0 {
        for chunk_y in min_chunk.1..=max_chunk.1 {
            for chunk_z in min_chunk.2..=max_chunk.2 {
                let chunk_coord = (chunk, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                let distance_squared = player_position
                    .translation
                    .distance_squared(chunk_world_pos);
                if distance_squared <= Z0_RADIUS_SQUARED {
                    if !chunk_svo.root.contains(&chunk_coord) {
                        if !chunks_being_loaded.0.contains_key(&chunk_coord) {
                            let request_id = chunks_being_loaded.1;
                            chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                            chunks_being_loaded.0.insert(chunk_coord, request_id);
                            let _ = chunk_channels.requests.send(ChunkRequest {
                                position: chunk_coord,
                                load_status: 0,
                                request_id,
                                upgrade: false,
                                distance_squared: distance_squared.round() as u32,
                            });
                        }
                    } else {
                        let load_status = chunk_svo.root.get(chunk_coord).unwrap().2;
                        if load_status == 1 {
                            if !chunks_being_loaded.0.contains_key(&chunk_coord) {
                                let request_id = chunks_being_loaded.1;
                                chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                                chunks_being_loaded.0.insert(chunk_coord, request_id);
                                let _ = chunk_channels.requests.send(ChunkRequest {
                                    position: chunk_coord,
                                    load_status: 0,
                                    request_id,
                                    upgrade: true,
                                    distance_squared: distance_squared as u32,
                                });
                            }
                        } else if load_status == 2 {
                            //this may need to change
                            if !chunks_being_loaded.0.contains_key(&chunk_coord) {
                                let request_id = chunks_being_loaded.1;
                                chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                                chunks_being_loaded.0.insert(chunk_coord, request_id);
                                let _ = chunk_channels.requests.send(ChunkRequest {
                                    position: chunk_coord,
                                    load_status: 0,
                                    request_id,
                                    upgrade: true,
                                    distance_squared: distance_squared as u32,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn validate_loading_queue(
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let player_position = player_transform.translation;
    chunks_being_loaded.0.retain(|chunk_coord, _request_id| {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        let distance_squared = player_position.distance_squared(chunk_world_pos);
        distance_squared <= Z2_RADIUS_SQUARED
    });
}
