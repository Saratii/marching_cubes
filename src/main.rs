use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::pbr::PbrPlugin;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::primitives::Aabb;
use bevy::render::view::NoFrustumCulling;
use bevy::window::PresentMode;
use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use marching_cubes::conversions::{
    chunk_coord_to_world_pos, world_pos_to_chunk_coord, world_pos_to_voxel_index,
};
use marching_cubes::data_loader::driver::{
    chunk_reciever, setup_loading_thread, validate_loading_queue,
};
use marching_cubes::data_loader::file_loader::{
    ChunkDataFileReadWrite, ChunkIndexMap, setup_chunk_loading, update_chunk_file_data,
};
use marching_cubes::marching_cubes::march_cubes;
use marching_cubes::player::player::{
    CameraController, KeyBindings, MainCameraTag, camera_look, camera_zoom, cursor_grab,
    initial_grab_cursor, player_movement, spawn_player, toggle_camera,
};
use marching_cubes::sparse_voxel_octree::ChunkSvo;
use marching_cubes::terrain::chunk_generator::{GenerateChunkEvent, LoadChunksEvent};
use marching_cubes::terrain::lod_zones::z2_chunk_load;
use marching_cubes::terrain::terrain::{
    CUBES_PER_CHUNK_DIM, ChunkTag, HALF_CHUNK, SDF_VALUES_PER_CHUNK_DIM,
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
        .add_event::<LoadChunksEvent>()
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
                setup_loading_thread.after(setup_chunk_loading),
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
                toggle_camera,
                camera_zoom,
                cursor_grab,
                camera_look,
                player_movement,
                z2_chunk_load,
                chunk_reciever.after(z2_chunk_load),
                validate_loading_queue.after(chunk_reciever),
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

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<MainCameraTag>>,
    window: Single<&Window>,
    mut svo: ResMut<ChunkSvo>,
    mut commands: Commands,
    mut dig_timer: Local<f32>,
    time: Res<Time>,
    chunk_data_file: Res<ChunkDataFileReadWrite>,
    chunk_index_map: Res<ChunkIndexMap>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    mut solid_chunk_query: Query<(&mut Collider, &mut Mesh3d, Entity), With<ChunkTag>>,
    mut mesh_handles: ResMut<Assets<Mesh>>,
) {
    const DIG_STRENGTH: f32 = 0.1;
    const DIG_TIMER: f32 = 0.02; // seconds
    const DIG_RADIUS: f32 = 0.3; // world space
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
                screen_to_world_ray(cursor_pos, camera, camera_transform, &svo)
            {
                let modified_chunks = svo.root.dig_sphere(world_pos, DIG_RADIUS, DIG_STRENGTH);
                if modified_chunks.is_empty() {
                    return;
                }
                for chunk_coord in modified_chunks {
                    let (entity, chunk) = svo.root.get_mut(chunk_coord).unwrap();
                    let new_mesh =
                        march_cubes(&chunk.sdfs, CUBES_PER_CHUNK_DIM, SDF_VALUES_PER_CHUNK_DIM);
                    let vertex_count = new_mesh.count_vertices();
                    if vertex_count > 0 {
                        let collider = Collider::from_bevy_mesh(
                            &new_mesh,
                            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                        )
                        .unwrap();
                        match entity {
                            //entity already existed, update it
                            Some(entity) => {
                                let (mut collider_component, mut mesh_handle, _) =
                                    solid_chunk_query.get_mut(*entity).unwrap();
                                *collider_component = collider;
                                mesh_handles.remove(&mesh_handle.0);
                                *mesh_handle = Mesh3d(mesh_handles.add(new_mesh));
                                commands.entity(*entity).insert(NoFrustumCulling); //this is bad
                            }
                            //entity did not already exist
                            None => {
                                let new_entity = commands
                                    .spawn((
                                        collider,
                                        Mesh3d(mesh_handles.add(new_mesh)),
                                        MeshMaterial3d(material_handle.0.clone()),
                                        Aabb {
                                            center: chunk_coord_to_world_pos(&chunk_coord).into(),
                                            half_extents: Vec3A::splat(HALF_CHUNK),
                                        },
                                        NoFrustumCulling, //this is bad
                                        ChunkTag,
                                        Transform::from_translation(chunk_coord_to_world_pos(
                                            &chunk_coord,
                                        )),
                                    ))
                                    .id();
                                *entity = Some(new_entity);
                            }
                        }
                    } else {
                        //no geometry, remove existing entity if it exists
                        if let Some(entity) = entity {
                            commands.entity(*entity).despawn();
                        }
                        *entity = None
                    }
                    let chunk_index_map = chunk_index_map.0.lock().unwrap();
                    let mut data_file = chunk_data_file.0.lock().unwrap();
                    update_chunk_file_data(&chunk_index_map, chunk_coord, &chunk, &mut data_file);
                    drop(chunk_index_map);
                    drop(data_file);
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
    svo: &ChunkSvo,
) -> Option<(Option<Entity>, Vec3, Vec3, (i16, i16, i16))> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 10.0;
    let step_size = 0.05;
    let mut distance_traveled = 0.0;
    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray_direction * distance_traveled;
        let chunk_coord = world_pos_to_chunk_coord(&current_pos);
        let chunk = svo.root.get(chunk_coord).unwrap();
        let voxel_idx = world_pos_to_voxel_index(&current_pos, &chunk_coord);
        let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
        if chunk.1.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
            return Some((chunk.0, current_pos, chunk_world_pos, chunk_coord));
        }
        distance_traveled += step_size;
    }
    None
}
