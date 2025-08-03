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
use fastnoise2::SafeNode;
use fastnoise2::generator::GeneratorWrapper;
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use marching_cubes::marching_cubes::march_cubes;
use marching_cubes::player::player::{
    CameraController, KeyBindings, PlayerTag, camera_look, camera_zoom, cursor_grab,
    initial_grab_cursor, player_movement, spawn_player, toggle_camera,
};
use marching_cubes::terrain::chunk_generator::{GenerateChunkEvent, NOISE_FREQUENCY, NOISE_SEED};
use marching_cubes::terrain::chunk_thread::{
    MyMapGenTasks, catch_chunk_generation_request, spawn_generated_chunks,
};
use marching_cubes::terrain::terrain::{
    CHUNK_CREATION_RADIUS, CHUNK_SIZE, ChunkMap, ChunkTag, Density, HALF_CHUNK, NoiseFunction,
    TerrainChunk, VOXEL_SIZE, VOXELS_PER_CHUNK, VOXELS_PER_DIM, setup_map,
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
                debug_listener,
                catch_chunk_generation_request,
                spawn_generated_chunks,
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
    chunk_map: ResMut<ChunkMap>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    perlin: Res<NoiseFunction>,
    mut chunk_generation_events: EventWriter<GenerateChunkEvent>,
    map_gen_tasks: Res<MyMapGenTasks>,
) {
    let player_chunk = ChunkMap::get_chunk_coord_from_world_pos(player_transform.translation);
    let radius = CHUNK_CREATION_RADIUS as f32;
    let radius_squared = radius * radius;
    let mut chunk_data = Vec::new();
    for dx in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
        for dz in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
            let xz_dist_sq = (dx * dx + dz * dz) as f32;
            if xz_dist_sq <= radius_squared {
                let max_dy = (radius_squared - xz_dist_sq).sqrt() as i16;
                let chunk_x = player_chunk.0 + dx;
                let chunk_z = player_chunk.2 + dz;
                let (min_height, max_height) =
                    get_chunk_column_height_range(chunk_x, chunk_z, &perlin.0);
                for dy in -max_dy..=max_dy {
                    let chunk_coord = (chunk_x, player_chunk.1 + dy, chunk_z);
                    if !chunk_map.0.contains_key(&chunk_coord)
                        && !map_gen_tasks.chunks_being_generated.contains(&chunk_coord)
                    {
                        let needs_noise = chunk_needs_noise(chunk_coord, min_height, max_height);
                        if needs_noise.is_some() {
                            chunk_data.push((chunk_coord, needs_noise.unwrap()));
                        }
                    }
                }
            }
        }
    }
    if !chunk_data.is_empty() {
        chunk_generation_events.write(GenerateChunkEvent { chunk_data });
    }
}

fn get_chunk_column_height_range(
    chunk_x: i16,
    chunk_z: i16,
    fbm: &GeneratorWrapper<SafeNode>,
) -> (f32, f32) {
    let chunk_world_x = chunk_x as f32 * CHUNK_SIZE;
    let chunk_world_z = chunk_z as f32 * CHUNK_SIZE;
    let half = HALF_CHUNK;
    let positions = [
        (chunk_world_x - half, chunk_world_z - half),
        (chunk_world_x + half, chunk_world_z - half),
        (chunk_world_x - half, chunk_world_z + half),
        (chunk_world_x + half, chunk_world_z + half),
        (chunk_world_x, chunk_world_z),
    ];
    let mut min_height = f32::INFINITY;
    let mut max_height = f32::NEG_INFINITY;
    for (x, z) in positions {
        let height = fbm.gen_single_2d(x * NOISE_FREQUENCY, z * NOISE_FREQUENCY, NOISE_SEED as i32);
        min_height = min_height.min(height);
        max_height = max_height.max(height);
    }
    (min_height, max_height)
}

fn chunk_needs_noise(
    chunk_coord: (i16, i16, i16),
    min_terrain_height: f32,
    max_terrain_height: f32,
) -> Option<bool> {
    let chunk_world_y = chunk_coord.1 as f32 * CHUNK_SIZE;
    let chunk_bottom = chunk_world_y - HALF_CHUNK;
    let chunk_top = chunk_world_y + HALF_CHUNK;
    let transition_width = 1.0;
    let effective_min = min_terrain_height - transition_width;
    let effective_max = max_terrain_height + transition_width;
    if chunk_bottom > effective_max {
        None
    } else if chunk_top < effective_min {
        Some(false)
    } else {
        Some(true)
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
) {
    const DIG_STRENGTH: f32 = 0.2;
    const DIG_TIMER: f32 = 0.05; // seconds
    const DIG_RADIUS: f32 = 0.2; // meters
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
            if let Ok((camera, camera_transform)) = camera_query.single() {
                if let Some((_, world_pos, _, _)) =
                    screen_to_world_ray(cursor_pos, camera, camera_transform, &chunk_map)
                {
                    let modified_chunks = chunk_map.dig_sphere(world_pos, DIG_RADIUS, DIG_STRENGTH);
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
) -> Option<(Entity, Vec3, Vec3, (i16, i16, i16))> {
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

fn debug_listener(
    keyboard: Res<ButtonInput<KeyCode>>,
    chunk_map: Res<ChunkMap>,
    my_tasks: Res<MyMapGenTasks>,
) {
    if keyboard.just_pressed(KeyCode::KeyF) {
        let map = &chunk_map.0;
        let struct_size = size_of_val(map);
        let mut vec_heap_size = 0;
        for (_, (_, chunk)) in map.iter() {
            vec_heap_size += chunk.densities.len() * size_of::<Density>();
        }
        println!("Size of a Density: {} bytes", size_of::<Density>());
        println!(
            "Densities per chunk: {} ({}^3)",
            VOXELS_PER_CHUNK, VOXELS_PER_DIM
        );
        let hashmap_heap =
            map.capacity() * (size_of::<(i16, i16, i16)>() + size_of::<(Entity, TerrainChunk)>());
        let total = struct_size + hashmap_heap + vec_heap_size;
        println!("Num Chunks: {}", map.len());
        println!("Tasks running: {}", my_tasks.generation_tasks.len());
        println!(
            "Chunks on generation thread: {}",
            my_tasks.chunks_being_generated.len()
        );
        println!("HashMap heap: {} bytes", hashmap_heap);
        println!("Vec heap: {} bytes", vec_heap_size);
        println!(
            "TOTAL: {} bytes ({:.2} MB)\n",
            total,
            total as f64 / 1_048_576.0
        );
    }
}
