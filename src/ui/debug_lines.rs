use std::collections::HashSet;

use bevy::prelude::*;
use bevy_rapier3d::prelude::Collider;

use crate::{
    constants::{CHUNK_WORLD_SIZE, CLUSTER_WORLD_LENGTH, REDUCED_LOD_1_RADIUS, REDUCED_LOD_2_RADIUS, REDUCED_LOD_3_RADIUS, REDUCED_LOD_4_RADIUS, REDUCED_LOD_5_RADIUS},
    conversions::{
        chunk_coord_to_cluster_coord, chunk_coord_to_world_pos, cluster_coord_to_world_center,
        world_pos_to_chunk_coord,
    },
    data_loader::driver::TerrainChunkMap,
    player::player::PlayerTag,
    terrain::terrain::ChunkTag,
    ui::configurable_settings::ConfigurableSettings,
};

pub fn draw_lod_debug(
    mut gizmos: Gizmos,
    player_transform_query: Query<&Transform, With<PlayerTag>>,
    settings: Res<ConfigurableSettings>,
) {
    let pos = player_transform_query.iter().next().unwrap().translation;
    if settings.debug_lod_1 {
        gizmos.sphere(pos, REDUCED_LOD_1_RADIUS, Color::srgb(1.0, 0.0, 0.0));
    }
    if settings.debug_lod_2 {
        gizmos.sphere(pos, REDUCED_LOD_2_RADIUS, Color::srgb(0.0, 1.0, 1.0));
    }
    if settings.debug_lod_3 {
        gizmos.sphere(pos, REDUCED_LOD_3_RADIUS, Color::srgb(0.0, 0.0, 1.0));
    }
    if settings.debug_lod_4 {
        gizmos.sphere(pos, REDUCED_LOD_4_RADIUS, Color::srgb(1.0, 1.0, 0.0));
    }
    if settings.debug_lod_5 {
        gizmos.sphere(pos, REDUCED_LOD_5_RADIUS, Color::srgb(1.0, 0.0, 1.0));
    }
}

pub fn draw_collider_debug(
    mut gizmos: Gizmos,
    query: Query<&Transform, (With<ChunkTag>, With<Collider>)>,
    terrain_chunk_map: Res<TerrainChunkMap>,
    settings: Res<ConfigurableSettings>,
) {
    if !settings.show_chunks {
        return;
    }
    for transform in query.iter() {
        gizmos.cube(
            Transform::from_translation(transform.translation)
                .with_scale(Vec3::splat(CHUNK_WORLD_SIZE)),
            Color::srgb(1.0, 0.0, 0.0),
        );
    }
    let terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
    for chunk_coord in terrain_chunk_map_lock.keys() {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        gizmos.cube(
            Transform::from_translation(chunk_world_pos)
                .with_scale(Vec3::splat(CHUNK_WORLD_SIZE as f32)),
            Color::srgba(0.0, 0.5, 1.0, 0.6),
        );
    }
}

pub fn draw_cluster_debug(
    mut gizmos: Gizmos,
    query: Query<&Transform, With<ChunkTag>>,
    settings: Res<ConfigurableSettings>,
) {
    if !settings.show_chunks {
        return;
    }
    let mut drawn_clusters = HashSet::new();
    for transform in query.iter() {
        let chunk_coord = world_pos_to_chunk_coord(&transform.translation);
        let cluster_coord = chunk_coord_to_cluster_coord(&chunk_coord);
        if drawn_clusters.insert(cluster_coord) {
            let cluster_world_pos = cluster_coord_to_world_center(&cluster_coord);
            gizmos.cube(
                Transform::from_translation(cluster_world_pos)
                    .with_scale(Vec3::splat(CLUSTER_WORLD_LENGTH)),
                Color::srgb(0.0, 1.0, 0.0),
            );
        }
    }
}
