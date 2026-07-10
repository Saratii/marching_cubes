use std::collections::HashSet;

use bevy::prelude::*;
use bevy_rapier3d::prelude::Collider;

use crate::{
    constants::{
        CHUNK_WORLD_SIZE, CLUSTER_WORLD_LENGTH, HALF_CHUNK, REDUCED_LOD_1_RADIUS,
        REDUCED_LOD_2_RADIUS, REDUCED_LOD_3_RADIUS, REDUCED_LOD_4_RADIUS, REDUCED_LOD_5_RADIUS,
        SAMPLES_PER_CHUNK_DIM, SAMPLES_PER_CHUNK_DIM_PADDED, VOXEL_WORLD_SIZE,
    },
    conversions::{
        chunk_coord_to_cluster_coord, chunk_coord_to_world_pos, cluster_coord_to_world_center,
        world_pos_to_chunk_coord,
    },
    deformable_terrain::{driver::TerrainChunkMap, plugin::ChunkTag, terrain::TerrainChunk},
    player::player::PlayerTag,
    ui::configurable_settings::ConfigurableSettings,
};

const CHUNKS_WITH_COLLIDER_COLOR: Color = Color::srgb(1.0, 0.0, 0.0);
const CHUNKS_IN_CHUNK_MAP_COLOR: Color = Color::srgba(0.0, 0.5, 1.0, 0.6);
const CLUSTER_COLOR: Color = Color::srgb(0.0, 1.0, 0.0);

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
            CHUNKS_WITH_COLLIDER_COLOR,
        );
    }
    let terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
    for chunk_coord in terrain_chunk_map_lock.keys() {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        gizmos.cube(
            Transform::from_translation(chunk_world_pos)
                .with_scale(Vec3::splat(CHUNK_WORLD_SIZE as f32)),
            CHUNKS_IN_CHUNK_MAP_COLOR,
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
                CLUSTER_COLOR,
            );
        }
    }
}

pub fn draw_voxel_surface_debug(
    mut gizmos: Gizmos,
    player_transform_query: Query<&Transform, With<PlayerTag>>,
    terrain_chunk_map: Res<TerrainChunkMap>,
    settings: Res<ConfigurableSettings>,
) {
    if !settings.show_voxels {
        return;
    }
    let player_pos = player_transform_query.iter().next().unwrap().translation;
    let chunk_coord = world_pos_to_chunk_coord(&player_pos);
    let map = terrain_chunk_map.0.lock().unwrap();
    let Some(TerrainChunk::NonUniformTerrainChunk(chunk)) = map.get(&chunk_coord) else {
        return;
    };
    let densities = &chunk.densities;
    let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
    let chunk_start = chunk_world_pos - Vec3::splat(HALF_CHUNK);
    for z in 1..=SAMPLES_PER_CHUNK_DIM {
        for y in 1..=SAMPLES_PER_CHUNK_DIM {
            for x in 1..=SAMPLES_PER_CHUNK_DIM {
                let idx = z * SAMPLES_PER_CHUNK_DIM_PADDED * SAMPLES_PER_CHUNK_DIM_PADDED
                    + y * SAMPLES_PER_CHUNK_DIM_PADDED
                    + x;
                let d = densities[idx];
                let neighbours = [
                    densities[idx + 1],
                    densities[idx - 1],
                    densities[idx + SAMPLES_PER_CHUNK_DIM_PADDED],
                    densities[idx - SAMPLES_PER_CHUNK_DIM_PADDED],
                    densities[idx + SAMPLES_PER_CHUNK_DIM_PADDED * SAMPLES_PER_CHUNK_DIM_PADDED],
                    densities[idx - SAMPLES_PER_CHUNK_DIM_PADDED * SAMPLES_PER_CHUNK_DIM_PADDED],
                ];
                if !neighbours.iter().any(|&n| (d >= 0) != (n >= 0)) {
                    continue;
                }
                let voxel_world_pos = chunk_start
                    + Vec3::new(
                        (x as f32 - 1.0) * VOXEL_WORLD_SIZE,
                        (y as f32 - 1.0) * VOXEL_WORLD_SIZE,
                        (z as f32 - 1.0) * VOXEL_WORLD_SIZE,
                    );
                let half_voxel = Vec3::splat(VOXEL_WORLD_SIZE * 0.5);
                let min = voxel_world_pos - half_voxel;
                let max = voxel_world_pos + half_voxel;
                let corners = [
                    Vec3::new(min.x, min.y, min.z),
                    Vec3::new(max.x, min.y, min.z),
                    Vec3::new(max.x, max.y, min.z),
                    Vec3::new(min.x, max.y, min.z),
                    Vec3::new(min.x, min.y, max.z),
                    Vec3::new(max.x, min.y, max.z),
                    Vec3::new(max.x, max.y, max.z),
                    Vec3::new(min.x, max.y, max.z),
                ];
                let color = Color::srgb(1.0, 1.0, 0.0);
                gizmos.line(corners[0], corners[1], color);
                gizmos.line(corners[1], corners[2], color);
                gizmos.line(corners[2], corners[3], color);
                gizmos.line(corners[3], corners[0], color);
                gizmos.line(corners[4], corners[5], color);
                gizmos.line(corners[5], corners[6], color);
                gizmos.line(corners[6], corners[7], color);
                gizmos.line(corners[7], corners[4], color);
                gizmos.line(corners[0], corners[4], color);
                gizmos.line(corners[1], corners[5], color);
                gizmos.line(corners[2], corners[6], color);
                gizmos.line(corners[3], corners[7], color);
            }
        }
    }
}
