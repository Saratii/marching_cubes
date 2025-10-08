use bevy::{
    ecs::{
        query::{With, Without},
        system::{Commands, ResMut, Single},
    },
    transform::components::Transform,
};

use crate::{
    data_loader::driver::{ChunkChannels, ChunksBeingLoaded},
    player::player::{MainCameraTag, PlayerTag},
    sparse_voxel_octree::ChunkSvo,
    terrain::terrain::Z2_RADIUS,
};

pub fn z2_chunk_load(
    mut svo: ResMut<ChunkSvo>,
    player_transform: Single<&mut Transform, (With<PlayerTag>, Without<MainCameraTag>)>,
    mut chunk_channels: ResMut<ChunkChannels>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    mut commands: Commands,
) {
    #[cfg(feature = "timers")]
    let start = std::time::Instant::now();
    let mut chunks_to_deallocate = Vec::new();
    svo.root.query_chunks_outside_sphere(
        &player_transform.translation,
        Z2_RADIUS,
        &mut chunks_to_deallocate,
    );
    for (chunk_coord, entity) in &chunks_to_deallocate {
        if let Some(canceled) = chunks_being_loaded.0.remove(chunk_coord) {
            canceled.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        svo.root.delete(*chunk_coord);
        if let Some(entity) = entity {
            commands.entity(*entity).despawn();
        }
    }
    svo.root.fill_missing_chunks_in_radius(
        &player_transform.translation,
        Z2_RADIUS,
        &mut chunks_being_loaded,
        &mut chunk_channels,
    );
    #[cfg(feature = "timers")]
    {
        let duration = start.elapsed();
        if duration > std::time::Duration::from_micros(200) {
            println!("{:<40} {:?}", "z2_chunk_load", duration);
        }
    }
}
