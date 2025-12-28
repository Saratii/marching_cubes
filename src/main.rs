use std::sync::atomic::Ordering;
use std::time::Duration;

use bevy::asset::UnapprovedPathMode;
use bevy::camera::primitives::MeshAabb;
use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSamplerDescriptor;
use bevy::pbr::{ExtendedMaterial, PbrPlugin};
use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy::winit::WinitSettings;
use bevy_rapier3d::plugin::{NoUserData, PhysicsSet, RapierPhysicsPlugin};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use bevy_volumetric_clouds::CloudsPlugin;
use fastnoise2::SafeNode;
use fastnoise2::generator::simplex::opensimplex2;
use fastnoise2::generator::{Generator, GeneratorWrapper};
// use bevy_rapier3d::render::RapierDebugRenderPlugin;
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use marching_cubes::conversions::{
    chunk_coord_to_world_pos, world_pos_to_chunk_coord, world_pos_to_voxel_index,
};
use marching_cubes::data_loader::driver::{
    INITIAL_CHUNKS_LOADED, TerrainChunkMap, WriteCmd, WriteCmdSender, chunk_spawn_reciever,
    project_downward, setup_chunk_driver,
};
use marching_cubes::data_loader::file_loader::{
    ChunkEntityMap, ChunkIndexMapDelta, ChunkIndexMapRead, setup_chunk_loading,
};
use marching_cubes::marching_cubes::mc::mc_mesh_generation;
use marching_cubes::player::player::{
    CameraController, KeyBindings, MainCameraTag, camera_look, camera_zoom, cursor_grab,
    initial_grab_cursor, player_movement, spawn_player, sync_player_mutex, toggle_camera,
};
use marching_cubes::sparse_voxel_octree::sphere_intersects_aabb;
use marching_cubes::terrain::chunk_generator::{dequantize_i16_to_f32, quantize_f32_to_i16};
use marching_cubes::terrain::terrain::{
    ChunkTag, HALF_CHUNK, NoiseFunction, SAMPLES_PER_CHUNK_DIM, TerrainChunk,
    TerrainMaterialHandle, UniformChunk, VOXEL_SIZE, generate_bevy_mesh, setup_map,
};
use marching_cubes::terrain::terrain_material::TerrainMaterialExtension;
use marching_cubes::ui::crosshair::spawn_crosshair;
use marching_cubes::ui::minimap::spawn_minimap;
use rayon::ThreadPoolBuilder;

fn main() {
    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    App::new()
        .insert_resource(KeyBindings::default())
        .insert_resource(CameraController::default())
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::reactive_low_power(Duration::from_secs_f64(
                1.0 / 240.0,
            )),
            unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(Duration::from_secs_f64(
                1.0 / 240.0,
            )),
        })
        .insert_resource(NoiseFunction(|| -> GeneratorWrapper<SafeNode> {
            (opensimplex2().fbm(1.0, 0.65, 3, 2.2)).build()
        }()))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin {
                    default_sampler: ImageSamplerDescriptor {
                        anisotropy_clamp: 16,
                        ..ImageSamplerDescriptor::linear()
                    },
                })
                .set(PbrPlugin { ..default() })
                .set(AssetPlugin {
                    unapproved_path_mode: UnapprovedPathMode::Allow, //because bevy is made by morons who don't know how CWD works
                    ..default()
                }),
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
            PerfUiPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>::default(
            ),
            CloudsPlugin,
            // RapierDebugRenderPlugin::default(),
        ))
        .add_systems(
            Startup,
            (
                setup_chunk_driver.after(spawn_player),
                setup,
                setup_chunk_loading,
                // generate_large_map_utility.after(setup_chunk_loading),
                setup_map,
                spawn_crosshair,
                spawn_player.after(setup_chunk_loading),
                spawn_minimap.after(spawn_player),
                initial_grab_cursor,
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
                sync_player_mutex.after(player_movement),
                chunk_spawn_reciever,
                project_downward
                    .after(PhysicsSet::SyncBackend)
                    .run_if(|| !INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed)),
            ),
        )
        .run();
}

fn setup(mut commands: Commands, mut ambient_light: ResMut<AmbientLight>) {
    commands.spawn(PerfUiDefaultEntries::default());
    commands.spawn((
        DirectionalLight {
            illuminance: 7000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            1.0,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
    ambient_light.brightness = 400.0;
}

pub fn dig_sphere(
    center: Vec3,
    radius: f32,
    strength: f32,
    terrain_chunk_map: &mut TerrainChunkMap,
) -> Vec<(i16, i16, i16)> {
    let mut modified_chunks = Vec::new();
    let min_world = center - Vec3::splat(radius);
    let max_world = center + Vec3::splat(radius);
    let min_chunk = world_pos_to_chunk_coord(&min_world);
    let max_chunk = world_pos_to_chunk_coord(&max_world);
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_y in min_chunk.1..=max_chunk.1 {
            for chunk_z in min_chunk.2..=max_chunk.2 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let mut terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
                let terrain_chunk = terrain_chunk_map_lock.get_mut(&chunk_coord).unwrap();
                let chunk_modified =
                    modify_chunk_voxels(terrain_chunk, chunk_coord, center, radius, strength);
                if chunk_modified && !modified_chunks.contains(&chunk_coord) {
                    modified_chunks.push(chunk_coord);
                }
            }
        }
    }
    modified_chunks
}

pub fn modify_chunk_voxels(
    terrain_chunk: &mut TerrainChunk,
    chunk_coord: (i16, i16, i16),
    dig_center: Vec3,
    radius: f32,
    strength: f32,
) -> bool {
    let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
    let chunk_world_size = SAMPLES_PER_CHUNK_DIM as f32 * VOXEL_SIZE;
    let node_min = Vec3::new(
        chunk_center.x - HALF_CHUNK,
        chunk_center.y - HALF_CHUNK,
        chunk_center.z - HALF_CHUNK,
    );
    let node_max = node_min + Vec3::splat(chunk_world_size);
    if !sphere_intersects_aabb(&dig_center, radius, &node_min, &node_max) {
        return false;
    }
    let mut chunk_modified = false;
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        for y in 0..SAMPLES_PER_CHUNK_DIM {
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let world_x = chunk_center.x - HALF_CHUNK + x as f32 * VOXEL_SIZE;
                let world_y = chunk_center.y - HALF_CHUNK + y as f32 * VOXEL_SIZE;
                let world_z = chunk_center.z - HALF_CHUNK + z as f32 * VOXEL_SIZE;
                let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                let distance = voxel_world_pos.distance(dig_center);
                if distance <= radius {
                    let falloff = 1.0 - (distance / radius).clamp(0.0, 1.0);
                    let dig_amount = strength * falloff;
                    let current_density =
                        terrain_chunk.get_mut_density(x as u32, y as u32, z as u32);
                    if *current_density < 0 {
                        let sdf_f32 = dequantize_i16_to_f32(*current_density);
                        let new_sdf = (sdf_f32 + dig_amount).clamp(-10.0, 10.0);
                        *current_density = quantize_f32_to_i16(new_sdf);
                        chunk_modified = true;
                    }
                }
            }
        }
    }
    return chunk_modified;
}

#[derive(SystemParam)]
struct TerrainIo<'w> {
    pub chunk_index_map_delta: Res<'w, ChunkIndexMapDelta>,
    pub chunk_index_map_read: Res<'w, ChunkIndexMapRead>,
    pub terrain_chunk_map: ResMut<'w, TerrainChunkMap>,
    pub chunk_entity_map: ResMut<'w, ChunkEntityMap>,
}

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCameraTag>>,
    window: Single<&Window>,
    mut commands: Commands,
    mut dig_timer: Local<f32>,
    time: Res<Time>,
    material_handle: Res<TerrainMaterialHandle>,
    mut solid_chunk_query: Query<(&mut Collider, &mut Mesh3d, Entity), With<ChunkTag>>,
    mut mesh_handles: ResMut<Assets<Mesh>>,
    mut terrain_io: TerrainIo,
    write_cmd_sender: Res<WriteCmdSender>,
) {
    const DIG_STRENGTH: f32 = 0.5;
    const DIG_TIMER: f32 = 0.02; // seconds
    const DIG_RADIUS: f32 = 1.0; // world space
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
            if let Some((world_pos, _, _)) = screen_to_world_ray(
                cursor_pos,
                camera.0,
                camera.1,
                &terrain_io.terrain_chunk_map,
            ) {
                let modified_chunks = dig_sphere(
                    world_pos,
                    DIG_RADIUS,
                    DIG_STRENGTH,
                    &mut terrain_io.terrain_chunk_map,
                );
                if modified_chunks.is_empty() {
                    return;
                }
                for chunk_coord in modified_chunks {
                    let entity = terrain_io.chunk_entity_map.0.get(&chunk_coord);
                    let mut terrain_chunk_map_lock = terrain_io.terrain_chunk_map.0.lock().unwrap();
                    let chunk = terrain_chunk_map_lock.get_mut(&chunk_coord).unwrap();
                    let chunk_clone = chunk.clone(); //clone instead of holding lock
                    match chunk_clone.is_uniform {
                        //this may need to also do the opposite, upgrading non-uniform to uniform
                        UniformChunk::Air | UniformChunk::Dirt => {
                            chunk.is_uniform = UniformChunk::NonUniform;
                            drop(terrain_chunk_map_lock);
                            write_cmd_sender
                                .0
                                .send(WriteCmd::WriteNonUniform {
                                    chunk: chunk_clone.clone(),
                                    coord: chunk_coord,
                                })
                                .unwrap();
                            if chunk_clone.is_uniform == UniformChunk::Air {
                                write_cmd_sender
                                    .0
                                    .send(WriteCmd::RemoveUniformAir { coord: chunk_coord })
                                    .unwrap();
                            } else if chunk_clone.is_uniform == UniformChunk::Dirt {
                                write_cmd_sender
                                    .0
                                    .send(WriteCmd::RemoveUniformDirt { coord: chunk_coord })
                                    .unwrap();
                            }
                        }
                        UniformChunk::NonUniform => {
                            drop(terrain_chunk_map_lock);
                        }
                    }
                    let (vertices, normals, material_ids, indices) = mc_mesh_generation(
                        &chunk_clone.densities,
                        &chunk_clone.materials,
                        SAMPLES_PER_CHUNK_DIM,
                        HALF_CHUNK,
                    );
                    let new_mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
                    if new_mesh.count_vertices() > 0 {
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
                                if let Some(aabb) = new_mesh.compute_aabb() {
                                    commands.entity(*entity).insert(aabb);
                                }
                                *mesh_handle = Mesh3d(mesh_handles.add(new_mesh));
                            }
                            //entity did not already exist
                            None => {
                                let new_entity = commands
                                    .spawn((
                                        collider,
                                        Mesh3d(mesh_handles.add(new_mesh)),
                                        MeshMaterial3d(material_handle.0.clone()),
                                        ChunkTag,
                                        Transform::from_translation(chunk_coord_to_world_pos(
                                            &chunk_coord,
                                        )),
                                    ))
                                    .id();
                                terrain_io
                                    .chunk_entity_map
                                    .0
                                    .insert(chunk_coord, new_entity);
                            }
                        }
                    } else {
                        //no geometry, remove existing entity if it exists
                        if let Some(entity) = entity {
                            commands.entity(*entity).despawn();
                        }
                        terrain_io.chunk_entity_map.0.remove(&chunk_coord);
                    }
                    //first check read only initial copy
                    let offset = {
                        let offset = terrain_io.chunk_index_map_read.0.get(&chunk_coord).cloned();
                        //if index isnt there maybe its in delta per chance
                        let offset = if offset.is_none() {
                            let index_map_lock = terrain_io.chunk_index_map_delta.0.lock().unwrap();
                            index_map_lock.get(&chunk_coord).cloned().clone()
                        } else {
                            offset
                        };
                        offset.unwrap()
                    };
                    write_cmd_sender
                        .0
                        .send(WriteCmd::UpdateNonUniform {
                            offset,
                            chunk: chunk_clone,
                        })
                        .unwrap();
                }
            }
        }
    }
}

fn screen_to_world_ray(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    terrain_chunk_map: &TerrainChunkMap,
) -> Option<(Vec3, Vec3, (i16, i16, i16))> {
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
        let chunk_coord = world_pos_to_chunk_coord(&current_pos);
        if let Some(chunk_data) = terrain_chunk_map.0.lock().unwrap().get(&chunk_coord) {
            let voxel_idx = world_pos_to_voxel_index(&current_pos, &chunk_coord);
            let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
            if chunk_data.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
                return Some((current_pos, chunk_world_pos, chunk_coord));
            }
            distance_traveled += step_size;
        } else {
            break;
        }
    }
    None
}
