use std::collections::HashSet;

use bevy::{
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
};
use bevy_rapier3d::prelude::Collider;

use crate::{
    conversions::{
        chunk_coord_to_cluster_coord, chunk_coord_to_world_pos, cluster_coord_to_world_center,
        world_pos_to_chunk_coord,
    },
    data_loader::driver::TerrainChunkMap,
    player::player::PlayerTag,
    terrain::terrain::{CHUNK_SIZE, CLUSTER_SIZE, ChunkTag, Z0_RADIUS, Z1_RADIUS},
};

#[derive(Component)]
pub struct DebugSphere1;

#[derive(Component)]
pub struct DebugSphere2;

pub fn spawn_debug_spheres(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere_1 = meshes.add(Sphere::new(Z0_RADIUS));
    let sphere_2 = meshes.add(Sphere::new(Z1_RADIUS));
    commands.spawn((
        DebugSphere1,
        Mesh3d(sphere_1.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.4, 0.8, 0.3),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        NotShadowCaster,
        NotShadowReceiver,
    ));

    commands.spawn((
        DebugSphere2,
        Mesh3d(sphere_2),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.8, 0.4, 0.4),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        NotShadowCaster,
        NotShadowReceiver,
    ));
}

pub fn update_debug_sphere_positions(
    player_transform: Single<&Transform, With<PlayerTag>>,
    mut sphere1_query: Query<&mut Transform, (With<DebugSphere1>, Without<PlayerTag>)>,
    mut sphere2_query: Query<
        &mut Transform,
        (
            With<DebugSphere2>,
            Without<PlayerTag>,
            Without<DebugSphere1>,
        ),
    >,
) {
    if let Ok(mut transform) = sphere1_query.single_mut() {
        transform.translation = player_transform.translation;
    }
    if let Ok(mut transform) = sphere2_query.single_mut() {
        transform.translation = player_transform.translation;
    }
}

pub fn draw_collider_debug(
    mut gizmos: Gizmos,
    query: Query<&Transform, (With<ChunkTag>, With<Collider>)>,
    terrain_chunk_map: Res<TerrainChunkMap>,
) {
    for transform in query.iter() {
        gizmos.cube(
            Transform::from_translation(transform.translation)
                .with_scale(Vec3::splat(CHUNK_SIZE as f32)),
            Color::srgb(1.0, 0.0, 0.0),
        );
    }
    let terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
    for chunk_coord in terrain_chunk_map_lock.keys() {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        gizmos.cube(
            Transform::from_translation(chunk_world_pos).with_scale(Vec3::splat(CHUNK_SIZE as f32)),
            Color::srgba(0.0, 0.5, 1.0, 0.6),
        );
    }
}

pub fn draw_cluster_debug(mut gizmos: Gizmos, query: Query<&Transform, With<ChunkTag>>) {
    let mut drawn_clusters = HashSet::new();
    for transform in query.iter() {
        let chunk_coord = world_pos_to_chunk_coord(&transform.translation);
        let cluster_coord = chunk_coord_to_cluster_coord(&chunk_coord);
        if drawn_clusters.insert(cluster_coord) {
            let cluster_world_pos = cluster_coord_to_world_center(&cluster_coord);
            gizmos.cube(
                Transform::from_translation(cluster_world_pos)
                    .with_scale(Vec3::splat(CHUNK_SIZE * CLUSTER_SIZE as f32)),
                Color::srgb(0.0, 1.0, 0.0),
            );
        }
    }
}
