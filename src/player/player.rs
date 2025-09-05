use std::{collections::HashMap, fs::File};

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};
use bevy_rapier3d::prelude::*;

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::chunk_loader::{ChunkDataFile, ChunkIndexMap, deallocate_chunks, load_chunk_data},
    marching_cubes::march_cubes,
    terrain::{
        chunk_generator::GenerateChunkEvent,
        chunk_thread::MyMapGenTasks,
        terrain::{
            CHUNK_CREATION_RADIUS, CHUNK_CREATION_RADIUS_SQUARED, CHUNK_SIZE, ChunkMap,
            StandardTerrainMaterialHandle, TerrainChunk, spawn_chunk,
        },
    },
};

const CAMERA_3RD_PERSON_OFFSET: Vec3 = Vec3 {
    x: 0.0,
    y: 5.0,
    z: 10.0,
};
const PLAYER_SPEED: f32 = 15.0;
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0., 20., 0.);
const PLAYER_CUBOID_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);
const CAMERA_FIRST_PERSON_OFFSET: Vec3 = Vec3::new(0., 0.75 * PLAYER_CUBOID_SIZE.y, 0.);
const MIN_ZOOM_DISTANCE: f32 = 5.0;
const MAX_ZOOM_DISTANCE: f32 = 50.0;
const ZOOM_SPEED: f32 = 5.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
const MIN_PITCH: f32 = -1.5;
const MAX_PITCH: f32 = 1.5;
const GRAVITY: f32 = -9.81 * 2.0;
const JUMP_IMPULSE: f32 = 7.0;

#[derive(Component)]
pub struct PlayerTag;

#[derive(Resource)]
pub struct CameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub is_first_person: bool,
    pub is_cursor_grabbed: bool,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.2,
            distance: CAMERA_3RD_PERSON_OFFSET.length(),
            is_first_person: true,
            is_cursor_grabbed: false,
        }
    }
}

#[derive(Component)]
pub struct VerticalVelocity {
    pub y: f32,
}

pub fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera_controller: Res<CameraController>,
) {
    let player_mesh = Cuboid::new(
        PLAYER_CUBOID_SIZE.x,
        PLAYER_CUBOID_SIZE.y,
        PLAYER_CUBOID_SIZE.z,
    );
    let player_mesh_handle = meshes.add(player_mesh);
    let material: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.3, 0.3, 0.1),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let yaw_rotation = Quat::from_rotation_y(camera_controller.yaw);
    let pitch_rotation = Quat::from_rotation_x(camera_controller.pitch);
    let initial_rotation = yaw_rotation * pitch_rotation;
    commands
        .spawn((
            Collider::cuboid(0.25, 0.75, 0.25),
            KinematicCharacterController {
                autostep: Some(CharacterAutostep {
                    max_height: CharacterLength::Absolute(0.1),
                    min_width: CharacterLength::Absolute(0.1),
                    include_dynamic_bodies: true,
                }),
                ..default()
            },
            Transform::from_translation(PLAYER_SPAWN),
            Mesh3d(player_mesh_handle),
            MeshMaterial3d(material),
            PlayerTag,
            VerticalVelocity { y: 0.0 },
        ))
        .with_child((
            Camera3d::default(),
            Transform {
                translation: CAMERA_FIRST_PERSON_OFFSET,
                rotation: initial_rotation,
                scale: Vec3::ONE,
            },
        ));
}

pub fn toggle_camera(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_transform: Single<&mut Transform, With<Camera3d>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if keyboard.just_pressed(KeyCode::KeyC) {
        if !camera_controller.is_first_person {
            camera_controller.is_first_person = true;
            camera_transform.translation = CAMERA_FIRST_PERSON_OFFSET;
            update_first_person_camera(&mut camera_transform, &camera_controller);
        } else {
            camera_controller.is_first_person = false;
            update_camera_position(&mut camera_transform, &camera_controller);
        }
    }
}

pub fn camera_zoom(
    mut scroll_events: EventReader<MouseWheel>,
    mut camera_transform: Single<&mut Transform, With<Camera3d>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if camera_controller.is_first_person || !camera_controller.is_cursor_grabbed {
        return;
    }
    for event in scroll_events.read() {
        let zoom_delta = event.y * ZOOM_SPEED;
        let current_distance = camera_transform.translation.length();
        let new_distance =
            (current_distance - zoom_delta).clamp(MIN_ZOOM_DISTANCE, MAX_ZOOM_DISTANCE);
        camera_controller.distance = new_distance;
        let zoom_factor = new_distance / current_distance;
        camera_transform.translation *= zoom_factor;
        update_camera_position(&mut camera_transform, &camera_controller)
    }
}

pub fn camera_look(
    mut mouse_motion: EventReader<MouseMotion>,
    mut camera_transform: Single<&mut Transform, With<Camera3d>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if camera_controller.is_cursor_grabbed {
        for event in mouse_motion.read() {
            camera_controller.yaw -= event.delta.x * MOUSE_SENSITIVITY;
            camera_controller.pitch -= event.delta.y * MOUSE_SENSITIVITY;
            camera_controller.pitch = camera_controller.pitch.clamp(MIN_PITCH, MAX_PITCH);
            if camera_controller.is_first_person {
                update_first_person_camera(&mut camera_transform, &camera_controller);
            } else {
                update_camera_position(&mut camera_transform, &camera_controller);
            }
        }
    }
}

fn update_camera_position(camera_transform: &mut Transform, controller: &CameraController) {
    let yaw_rotation = Quat::from_rotation_y(controller.yaw);
    let pitch_rotation = Quat::from_rotation_x(controller.pitch);
    let rotation = yaw_rotation * pitch_rotation;
    let offset = rotation * Vec3::new(0.0, 0.0, controller.distance);
    camera_transform.translation = offset;
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
}

fn update_first_person_camera(camera_transform: &mut Transform, controller: &CameraController) {
    let yaw_rotation = Quat::from_rotation_y(controller.yaw);
    let pitch_rotation = Quat::from_rotation_x(controller.pitch);
    let rotation = yaw_rotation * pitch_rotation;
    camera_transform.rotation = rotation;
}

fn toggle_grab_cursor(window: &mut Window, camera_controller: &mut CameraController) {
    match window.cursor_options.grab_mode {
        CursorGrabMode::None => {
            window.cursor_options.grab_mode = CursorGrabMode::Confined;
            window.cursor_options.visible = false;
            camera_controller.is_cursor_grabbed = true;
        }
        _ => {
            window.cursor_options.grab_mode = CursorGrabMode::None;
            window.cursor_options.visible = true;
            camera_controller.is_cursor_grabbed = false;
        }
    }
}

pub fn initial_grab_cursor(
    mut primary_window: Single<&mut Window, With<PrimaryWindow>>,
    mut camera_controller: ResMut<CameraController>,
) {
    toggle_grab_cursor(&mut primary_window, &mut camera_controller);
}

pub fn cursor_grab(
    keys: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    mut primary_window: Single<&mut Window, With<PrimaryWindow>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if keys.just_pressed(key_bindings.toggle_grab_cursor) {
        toggle_grab_cursor(&mut primary_window, &mut camera_controller);
    }
}

#[derive(Resource)]
pub struct KeyBindings {
    pub move_forward: KeyCode,
    pub move_backward: KeyCode,
    pub move_left: KeyCode,
    pub move_right: KeyCode,
    pub jump: KeyCode,
    pub toggle_grab_cursor: KeyCode,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            move_forward: KeyCode::KeyW,
            move_backward: KeyCode::KeyS,
            move_left: KeyCode::KeyA,
            move_right: KeyCode::KeyD,
            jump: KeyCode::Space,
            toggle_grab_cursor: KeyCode::Escape,
        }
    }
}

pub fn player_movement(
    time: Res<Time>,
    mut player_query: Query<
        (
            &mut KinematicCharacterController,
            &mut VerticalVelocity,
            Option<&KinematicCharacterControllerOutput>,
        ),
        With<PlayerTag>,
    >,
    keyboard: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    camera_controller: Res<CameraController>,
) {
    for (mut controller, mut vertical_velocity, controller_output) in player_query.iter_mut() {
        let is_grounded = controller_output.map_or(false, |output| output.grounded);
        let mut movement_vec = Vec3::ZERO;
        let yaw_rotation = Quat::from_rotation_y(camera_controller.yaw);
        let pitch_rotation = Quat::from_rotation_x(camera_controller.pitch);
        let camera_rotation = yaw_rotation * pitch_rotation;
        let forward = camera_rotation * Vec3::NEG_Z;
        let right = camera_rotation * Vec3::X;
        if keyboard.pressed(key_bindings.move_forward) {
            movement_vec += forward * PLAYER_SPEED;
        }
        if keyboard.pressed(key_bindings.move_backward) {
            movement_vec -= forward * PLAYER_SPEED;
        }
        if keyboard.pressed(key_bindings.move_left) {
            movement_vec -= right * PLAYER_SPEED;
        }
        if keyboard.pressed(key_bindings.move_right) {
            movement_vec += right * PLAYER_SPEED;
        }
        if keyboard.just_pressed(key_bindings.jump) && is_grounded {
            vertical_velocity.y = JUMP_IMPULSE;
        }
        if !is_grounded {
            vertical_velocity.y += GRAVITY * time.delta_secs();
        } else if vertical_velocity.y < 0.0 {
            vertical_velocity.y = 0.0;
        }
        movement_vec.y = vertical_velocity.y;
        controller.translation = Some(movement_vec * time.delta_secs());
    }
}

pub fn detect_chunk_border_crossing(
    player_transform: Single<&Transform, (With<PlayerTag>, Changed<Transform>)>,
    mut last_chunk: Local<Option<(i16, i16, i16)>>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut chunk_generation_events: EventWriter<GenerateChunkEvent>,
    mut map_gen_tasks: ResMut<MyMapGenTasks>,
    chunk_index_map: Res<ChunkIndexMap>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_terrain_material_handle: Res<StandardTerrainMaterialHandle>,
    mut last_position_of_load: Local<Option<Vec3>>,
) {
    const GRACE_RADIUS: f32 = 10.0; //distance from last load where movement will not trigger a reload.
    let current_chunk = world_pos_to_chunk_coord(&player_transform.translation);
    if last_chunk.is_none() {
        *last_chunk = Some(current_chunk);
        return;
    }
    if current_chunk != last_chunk.unwrap() {
        deallocate_chunks(current_chunk, &mut chunk_map.0, &mut commands);
        *last_chunk = Some(current_chunk);
        if last_position_of_load.is_none() {
            *last_position_of_load = Some(player_transform.translation);
        } else if player_transform
            .translation
            .distance_squared(last_position_of_load.unwrap())
            < GRACE_RADIUS * GRACE_RADIUS
        {
            return;
        } else {
            *last_position_of_load = Some(player_transform.translation);
        }
        update_chunks(
            &mut chunk_map.0,
            &player_transform.translation,
            &mut chunk_generation_events,
            &mut map_gen_tasks,
            &chunk_index_map.0,
            &mut chunk_data_file.0,
            &mut commands,
            &mut meshes,
            &standard_terrain_material_handle.0,
            &current_chunk,
        );
    }
}

fn update_chunks(
    chunk_map: &mut HashMap<(i16, i16, i16), (Entity, TerrainChunk)>,
    player_translation: &Vec3,
    chunk_generation_events: &mut EventWriter<GenerateChunkEvent>,
    map_gen_tasks: &mut MyMapGenTasks,
    chunk_index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_data_file: &mut File,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    standard_terrain_material_handle: &Handle<StandardMaterial>,
    player_chunk: &(i16, i16, i16),
) {
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
            let dx = (chunk_x as f32 * CHUNK_SIZE) - player_translation.x;
            let dz = (chunk_z as f32 * CHUNK_SIZE) - player_translation.z;
            if dx * dx + dz * dz > CHUNK_CREATION_RADIUS_SQUARED {
                continue; // skip third dimension of chunk loop if too far in xz plane
            }
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                if chunk_world_pos.distance_squared(*player_translation)
                    < CHUNK_CREATION_RADIUS_SQUARED
                {
                    if !chunk_map.contains_key(&chunk_coord)
                        && !map_gen_tasks.chunks_being_generated.contains(&chunk_coord)
                    {
                        if chunk_index_map.contains_key(&chunk_coord) {
                            let chunk_data =
                                load_chunk_data(chunk_data_file, chunk_index_map, chunk_coord);
                            let mesh: Mesh = march_cubes(&chunk_data.densities);
                            let collider = if mesh.count_vertices() > 0 {
                                Collider::from_bevy_mesh(
                                    &mesh,
                                    &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                                )
                            } else {
                                None
                            };
                            let entity = spawn_chunk(
                                commands,
                                meshes,
                                standard_terrain_material_handle.clone(),
                                mesh,
                                Transform::from_translation(chunk_world_pos),
                                collider,
                            );
                            chunk_map.insert(chunk_coord, (entity, chunk_data));
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
