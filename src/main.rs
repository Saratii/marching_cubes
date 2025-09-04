use std::collections::HashSet;

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
use marching_cubes::conversions::{
    chunk_coord_to_world_pos, world_pos_to_chunk_coord, world_pos_to_voxel_index,
};
use marching_cubes::data_loader::chunk_loader::{
    ChunkDataFile, ChunkIndexMap, load_chunk_data, setup_chunk_loading, update_chunk_file_data,
};
use marching_cubes::marching_cubes::march_cubes;
use marching_cubes::player::player::{
    CameraController, KeyBindings, PlayerTag, camera_look, camera_zoom, cursor_grab,
    detect_chunk_border_crossing, initial_grab_cursor, player_movement, spawn_player,
    toggle_camera,
};
use marching_cubes::terrain::chunk_generator::GenerateChunkEvent;
use marching_cubes::terrain::chunk_thread::{
    MyMapGenTasks, catch_chunk_generation_request, spawn_generated_chunks,
};
use marching_cubes::terrain::terrain::{
    CHUNK_CREATION_RADIUS, CHUNK_CREATION_RADIUS_SQUARED, ChunkMap, ChunkTag, HALF_CHUNK,
    StandardTerrainMaterialHandle, setup_map, spawn_initial_chunks,
};
use rayon::ThreadPoolBuilder;

fn main() {
    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    App::new()
        .insert_resource(KeyBindings::default())
        .insert_resource(CameraController::default())
        .insert_resource(MyMapGenTasks {
            generation_tasks: Vec::new(),
            chunks_being_generated: HashSet::new(),
        })
        .add_event::<GenerateChunkEvent>()
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
                spawn_initial_chunks.after(setup_chunk_loading),
                setup_chunk_loading,
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
                player_movement,
                catch_chunk_generation_request,
                spawn_generated_chunks,
                detect_chunk_border_crossing,
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
    mut chunk_map: ResMut<ChunkMap>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    mut chunk_generation_events: EventWriter<GenerateChunkEvent>,
    mut map_gen_tasks: ResMut<MyMapGenTasks>,
    chunk_index_map: Res<ChunkIndexMap>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_terrain_material_handle: Res<StandardTerrainMaterialHandle>,
) {
    let player_chunk = world_pos_to_chunk_coord(player_transform.translation);
    let mut chunk_coords = Vec::new();
    let min_chunk = (
        player_chunk.0 - CHUNK_CREATION_RADIUS as i16,
        player_chunk.1 - CHUNK_CREATION_RADIUS as i16,
        player_chunk.2 - CHUNK_CREATION_RADIUS as i16,
    );
    let max_chunk = (
        player_chunk.0 + CHUNK_CREATION_RADIUS as i16,
        player_chunk.1 + CHUNK_CREATION_RADIUS as i16,
        player_chunk.2 + CHUNK_CREATION_RADIUS as i16,
    );
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
                if chunk_world_pos.distance_squared(player_transform.translation)
                    < CHUNK_CREATION_RADIUS_SQUARED
                {
                    if !chunk_map.0.contains_key(&chunk_coord)
                        && !map_gen_tasks.chunks_being_generated.contains(&chunk_coord)
                    {
                        if chunk_index_map.0.contains_key(&chunk_coord) {
                            let chunk_data = load_chunk_data(
                                &mut chunk_data_file.0,
                                &chunk_index_map.0,
                                chunk_coord,
                            );
                            let mesh = march_cubes(&chunk_data.densities);
                            let transform =
                                Transform::from_translation(chunk_coord_to_world_pos(chunk_coord));
                            let collider = if mesh.count_vertices() > 0 {
                                Collider::from_bevy_mesh(
                                    &mesh,
                                    &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                                )
                            } else {
                                None
                            };
                            let entity = chunk_map.spawn_chunk(
                                &mut commands,
                                &mut meshes,
                                standard_terrain_material_handle.0.clone(),
                                mesh,
                                transform,
                                collider,
                            );
                            chunk_map.0.insert(chunk_coord, (entity, chunk_data));
                        } else {
                            chunk_coords.push(chunk_coord);
                            map_gen_tasks.chunks_being_generated.insert(chunk_coord);
                        }
                    }
                }
            }
        }
    }
    if !chunk_coords.is_empty() {
        chunk_generation_events.write(GenerateChunkEvent { chunk_coords });
    }
}

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut chunk_mesh_query: Query<(Entity, &mut Mesh3d), With<ChunkTag>>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera>>,
    window: Single<&Window>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut dig_timer: Local<f32>,
    time: Res<Time>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
    chunk_index_map: Res<ChunkIndexMap>,
) {
    const DIG_STRENGTH: f32 = 2.2;
    const DIG_TIMER: f32 = 0.02; // seconds
    const DIG_RADIUS: f32 = 2.2; // world space
    let should_dig = if mouse_input.pressed(MouseButton::Left) {
        *dig_timer += time.delta_secs();
        if *dig_timer >= DIG_TIMER {
            *dig_timer = 0.0;
            true
        } else {
            false
        }
    } else {
        *dig_timer = 0.0;
        false
    };
    if should_dig {
        if let Some(cursor_pos) = window.cursor_position() {
            let (camera, camera_transform) = camera_query.single().unwrap();
            if let Some((_, world_pos, _, _)) =
                screen_to_world_ray(cursor_pos, camera, camera_transform, &chunk_map)
            {
                let modified_chunks = chunk_map.dig_sphere(world_pos, DIG_RADIUS, DIG_STRENGTH);
                if modified_chunks.is_empty() {
                    return;
                }
                for chunk_coord in modified_chunks {
                    if let Some((entity, chunk)) = chunk_map.0.get(&chunk_coord) {
                        if let Ok((_, mut mesh_handle)) = chunk_mesh_query.get_mut(*entity) {
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
                        update_chunk_file_data(
                            &chunk_index_map.0,
                            chunk_coord,
                            &chunk,
                            &mut chunk_data_file.0,
                        );
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
) -> Option<(Entity, Vec3, Vec3, (i16, i16, i16))> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 8.0;
    let step_size = 0.05;
    let mut distance_traveled = 0.0;
    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray_direction * distance_traveled;
        let chunk_coord = world_pos_to_chunk_coord(current_pos);
        if let Some(chunk) = chunk_map.0.get(&chunk_coord) {
            let voxel_idx = world_pos_to_voxel_index(current_pos, chunk_coord);
            let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
            if chunk.1.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
                return Some((chunk.0, current_pos, chunk_world_pos, chunk_coord));
            }
        }
        distance_traveled += step_size;
    }
    None
}
