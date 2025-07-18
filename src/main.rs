use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::pbr::PbrPlugin;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::mesh::MeshAabb;
use bevy::render::primitives::Aabb;
use bevy::window::PresentMode;
use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use marching_cubes::marching_cubes::{HALF_CHUNK, march_cubes};
use marching_cubes::player::player::{
    CameraController, KeyBindings, PlayerTag, camera_look, camera_zoom, cursor_grab,
    initial_grab_cursor, spawn_player, toggle_camera,
};
use marching_cubes::terrain_generation::{
    CHUNK_SIZE, ChunkMap, ChunkTag, NoiseFunction, StandardTerrainMaterialHandle, VOXEL_SIZE,
    setup_map,
};

pub const CHUNK_CREATION_RADIUS: i32 = 10; //in chunks
pub const CHUNK_GENERATION_CIRCULAR_RADIUS_SQUARED: f32 =
    (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE) * (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE);

fn main() {
    App::new()
        .insert_resource(KeyBindings::default())
        .insert_resource(CameraController::default())
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(PbrPlugin { ..default() }),
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin,
            RenderDiagnosticsPlugin,
            SystemInformationDiagnosticsPlugin,
            PerfUiPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
        ))
        .insert_resource(ClearColor(Color::srgb(0.0, 1.0, 1.0)))
        .add_systems(
            Startup,
            (
                setup,
                setup_map,
                setup_crosshair,
                spawn_player,
                initial_grab_cursor,
            ),
        )
        .add_systems(
            Update,
            (
                handle_digging_input,
                update_chunks,
                toggle_camera,
                camera_zoom,
                camera_look,
                cursor_grab,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(PerfUiDefaultEntries::default());
    commands.spawn((
        DirectionalLight { ..default() },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            1.0,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
}

//this should ideally only trigger when the player moves across chunk borders
fn update_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_map: ResMut<ChunkMap>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    perlin: Res<NoiseFunction>,
    standard_terrain_material_handle: Res<StandardTerrainMaterialHandle>,
) {
    let player_chunk = ChunkMap::get_chunk_coord_from_world_pos(player_transform.translation);
    let radius = CHUNK_CREATION_RADIUS as f32;
    let radius_squared = radius * radius;
    for dx in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
        for dz in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
            let xz_dist_sq = (dx * dx + dz * dz) as f32;
            if xz_dist_sq <= radius_squared {
                let max_dy = (radius_squared - xz_dist_sq).sqrt() as i32;
                for dy in -max_dy..=max_dy {
                    let chunk_coord = (
                        player_chunk.0 + dx,
                        player_chunk.1 + dy,
                        player_chunk.2 + dz,
                    );
                    if !chunk_map.0.contains_key(&chunk_coord) {
                        let entity = chunk_map.spawn_chunk(
                            &mut commands,
                            &mut meshes,
                            chunk_coord,
                            &perlin.0,
                            &standard_terrain_material_handle.0,
                        );
                        chunk_map.0.insert(chunk_coord, entity);
                    }
                }
            }
        }
    }
}

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut chunk_mesh_query: Query<(Entity, &mut Mesh3d), With<ChunkTag>>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera>>,
    windows: Query<&Window>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
) {
    if mouse_input.just_pressed(MouseButton::Left) {
        if let Ok(window) = windows.single() {
            if let Some(cursor_pos) = window.cursor_position() {
                if let Ok((camera, camera_transform)) = camera_query.single() {
                    if let Some((_, world_pos, _, _)) =
                        screen_to_world_ray(cursor_pos, camera, camera_transform, &chunk_map)
                    {
                        let modified_chunks = chunk_map.dig_sphere(world_pos, 2.0, 5.0);
                        for chunk_coord in modified_chunks {
                            if let Some((entity, chunk)) = chunk_map.0.get(&chunk_coord) {
                                if let Ok((_, mut mesh_handle)) = chunk_mesh_query.get_mut(*entity)
                                {
                                    let new_mesh = march_cubes(&chunk.densities);
                                    if let Some(_) = new_mesh.compute_aabb() {
                                        let min = Vec3::new(-HALF_CHUNK, -HALF_CHUNK, -HALF_CHUNK);
                                        let max = Vec3::new(HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
                                        let center = (min + max) / 2.0;
                                        let half_extents = (max - min) / 2.0;
                                        let expanded_aabb = Aabb {
                                            center: center.into(),
                                            half_extents: half_extents.into(),
                                        };
                                        commands.entity(*entity).insert(expanded_aabb);
                                        commands.entity(*entity).remove::<Collider>();
                                        if new_mesh.count_vertices() > 0 {
                                            commands.entity(*entity).insert(
                                                Collider::from_bevy_mesh(
                                                    &new_mesh,
                                                    &ComputedColliderShape::TriMesh(
                                                        TriMeshFlags::default(),
                                                    ),
                                                )
                                                .unwrap(),
                                            );
                                        }
                                    }
                                    *mesh_handle = Mesh3d(meshes.add(new_mesh));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn setup_crosshair(mut commands: Commands) {
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            position_type: PositionType::Absolute,
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(Node {
                    width: Val::Px(20.0),
                    height: Val::Px(20.0),
                    position_type: PositionType::Relative,
                    ..default()
                })
                .with_children(|crosshair| {
                    crosshair.spawn((
                        Node {
                            width: Val::Px(20.0),
                            height: Val::Px(2.0),
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(9.0),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                    crosshair.spawn((
                        Node {
                            width: Val::Px(2.0),
                            height: Val::Px(20.0),
                            position_type: PositionType::Absolute,
                            left: Val::Px(9.0),
                            top: Val::Px(0.0),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                });
        });
}

fn screen_to_world_ray(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    chunk_map: &ResMut<ChunkMap>,
) -> Option<(Entity, Vec3, Vec3, (i32, i32, i32))> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 100.0;
    let step_size = 0.1;
    let mut distance_traveled = 0.0;
    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray_direction * distance_traveled;
        let chunk_coord = ChunkMap::get_chunk_coord_from_world_pos(current_pos);
        if let Some(chunk) = chunk_map.0.get(&chunk_coord) {
            let local_pos = current_pos - chunk.1.world_position;
            let voxel_x = ((local_pos.x + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
            let voxel_y = ((local_pos.y + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
            let voxel_z = ((local_pos.z + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
            if chunk.1.is_solid(voxel_x, voxel_y, voxel_z) {
                return Some((chunk.0, current_pos, chunk.1.world_position, chunk_coord));
            }
            distance_traveled += step_size;
        } else {
            return None;
        }
    }
    None
}
