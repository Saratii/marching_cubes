use std::sync::{Arc, Mutex, atomic::Ordering};

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_rapier3d::prelude::*;

use crate::{
    data_loader::{
        driver::{INITIAL_CHUNKS_LOADED, PlayerTranslationMutexHandle},
        file_loader::{PlayerDataFile, read_player_position, write_player_position},
    },
    terrain::{chunk_generator::sample_fbm, terrain::NoiseFunction},
};

const CAMERA_3RD_PERSON_OFFSET: Vec3 = Vec3 {
    x: 0.0,
    y: 5.0,
    z: 10.0,
};
const PLAYER_SPEED: f32 = 5.0;
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0., 0., 0.);
const PLAYER_CUBOID_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);
pub const CAMERA_FIRST_PERSON_OFFSET: Vec3 = Vec3::new(0., 0.75 * PLAYER_CUBOID_SIZE.y, 0.);
const MIN_ZOOM_DISTANCE: f32 = 4.0;
const MAX_ZOOM_DISTANCE: f32 = 2000.0;
const MIN_ZOOM_SPEED: f32 = 0.5;
const MAX_ZOOM_SPEED: f32 = 120.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
const MIN_PITCH: f32 = -1.5;
const MAX_PITCH: f32 = 1.5;
const BASE_GRAVITY: f32 = -9.81;
const JUMP_IMPULSE: f32 = 7.0;

#[derive(Component)]
pub struct PlayerTag;

#[derive(Component)]
pub struct VerticalVelocity {
    pub y: f32,
}

#[derive(Component)]
pub struct MainCameraTag;

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

pub fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut player_data_file: ResMut<PlayerDataFile>,
    fbm: Res<NoiseFunction>,
    main_camera: Single<Entity, With<MainCameraTag>>,
) {
    let player_spawn = match read_player_position(&mut player_data_file.0) {
        Some(pos) => pos,
        None => Vec3::new(
            PLAYER_SPAWN.x,
            sample_fbm(&fbm.0, PLAYER_SPAWN.x, PLAYER_SPAWN.z) + 20.0,
            PLAYER_SPAWN.z,
        ),
    };
    println!("Spawning player at position: {:?}", player_spawn);
    commands.insert_resource(PlayerTranslationMutexHandle(Arc::new(Mutex::new(
        player_spawn,
    ))));
    let player_mesh = Cuboid::new(
        PLAYER_CUBOID_SIZE.x,
        PLAYER_CUBOID_SIZE.y,
        PLAYER_CUBOID_SIZE.z,
    );
    let player_mesh_handle = meshes.add(player_mesh);
    let material: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.3, 0.3, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let player = commands
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
            Transform::from_translation(player_spawn),
            Mesh3d(player_mesh_handle),
            MeshMaterial3d(material),
            PlayerTag,
            VerticalVelocity { y: 0.0 },
            Visibility::Hidden,
        ))
        .id();
    commands.entity(player).add_child(*main_camera);
}

pub fn toggle_camera(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_transform: Single<&mut Transform, With<MainCameraTag>>,
    mut camera_controller: ResMut<CameraController>,
    mut player_visibility: Single<&mut Visibility, With<PlayerTag>>,
) {
    if keyboard.just_pressed(KeyCode::KeyC) {
        if !camera_controller.is_first_person {
            camera_controller.is_first_person = true;
            camera_transform.translation = CAMERA_FIRST_PERSON_OFFSET;
            update_first_person_camera(&mut camera_transform, &camera_controller);
            **player_visibility = Visibility::Hidden;
        } else {
            camera_controller.is_first_person = false;
            update_camera_position(&mut camera_transform, &camera_controller);
            **player_visibility = Visibility::Visible;
        }
    }
}

pub fn camera_zoom(
    mut scroll_events: MessageReader<MouseWheel>,
    mut camera_transform: Single<&mut Transform, With<MainCameraTag>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if camera_controller.is_first_person || !camera_controller.is_cursor_grabbed {
        return;
    }
    for event in scroll_events.read() {
        let current_distance = camera_transform.translation.length();
        let t = (current_distance - MIN_ZOOM_DISTANCE) / (MAX_ZOOM_DISTANCE - MIN_ZOOM_DISTANCE);
        let zoom_speed = MIN_ZOOM_SPEED + t * (MAX_ZOOM_SPEED - MIN_ZOOM_SPEED);
        let zoom_delta = event.y * zoom_speed;
        let new_distance =
            (current_distance - zoom_delta).clamp(MIN_ZOOM_DISTANCE, MAX_ZOOM_DISTANCE);
        camera_controller.distance = new_distance;
        let zoom_factor = new_distance / current_distance;
        camera_transform.translation *= zoom_factor;
        update_camera_position(&mut camera_transform, &camera_controller)
    }
}

pub fn camera_look(
    mut mouse_motion: MessageReader<MouseMotion>,
    mut camera_transform: Single<&mut Transform, With<MainCameraTag>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if camera_controller.is_cursor_grabbed {
        let mut angles_changed = false;
        for event in mouse_motion.read() {
            let old_yaw = camera_controller.yaw;
            let old_pitch = camera_controller.pitch;
            camera_controller.yaw -= event.delta.x * MOUSE_SENSITIVITY;
            camera_controller.pitch -= event.delta.y * MOUSE_SENSITIVITY;
            camera_controller.pitch = camera_controller.pitch.clamp(MIN_PITCH, MAX_PITCH);
            if camera_controller.yaw != old_yaw || camera_controller.pitch != old_pitch {
                angles_changed = true;
            }
        }
        if angles_changed {
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

fn toggle_grab_cursor(
    primary_cursor_options: &mut CursorOptions,
    camera_controller: &mut CameraController,
) {
    match primary_cursor_options.grab_mode {
        CursorGrabMode::None => {
            primary_cursor_options.grab_mode = CursorGrabMode::Confined;
            primary_cursor_options.visible = false;
            camera_controller.is_cursor_grabbed = true;
        }
        _ => {
            primary_cursor_options.grab_mode = CursorGrabMode::None;
            primary_cursor_options.visible = true;
            camera_controller.is_cursor_grabbed = false;
        }
    }
}

pub fn initial_grab_cursor(
    mut primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera_controller: ResMut<CameraController>,
) {
    toggle_grab_cursor(&mut primary_cursor_options, &mut camera_controller);
}

pub fn cursor_grab(
    keys: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    mut primary_cursor_options: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera_controller: ResMut<CameraController>,
) {
    if keys.just_pressed(key_bindings.toggle_grab_cursor) {
        toggle_grab_cursor(&mut primary_cursor_options, &mut camera_controller);
    }
}

pub fn player_movement(
    time: Res<Time>,
    mut player: Single<
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
    let is_grounded = player.2.map_or(false, |output| output.grounded);
    let mut movement_vec = Vec3::ZERO;
    let yaw_rotation = Quat::from_rotation_y(camera_controller.yaw);
    let forward = yaw_rotation * Vec3::NEG_Z;
    let right = yaw_rotation * Vec3::X;
    let mut horizontal_movement = Vec3::ZERO;
    if keyboard.pressed(key_bindings.move_forward) {
        horizontal_movement += forward;
    }
    if keyboard.pressed(key_bindings.move_backward) {
        horizontal_movement -= forward;
    }
    if keyboard.pressed(key_bindings.move_left) {
        horizontal_movement -= right;
    }
    if keyboard.pressed(key_bindings.move_right) {
        horizontal_movement += right;
    }
    if horizontal_movement.length() > 0.0 {
        horizontal_movement = horizontal_movement.normalize();
        movement_vec += horizontal_movement * PLAYER_SPEED;
    }
    if keyboard.just_pressed(key_bindings.jump) && is_grounded {
        player.1.y = JUMP_IMPULSE;
    }
    if !is_grounded {
        player.1.y += BASE_GRAVITY
            * time.delta_secs()
            * INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed) as u8 as f32;
    } else if player.1.y < 0.0 {
        player.1.y = 0.0;
    }
    movement_vec.y = player.1.y;
    player.0.translation = Some(movement_vec * time.delta_secs());
}

pub fn sync_player_mutex(
    player_transform_mutex_handle: ResMut<PlayerTranslationMutexHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    mut player_data_file: ResMut<PlayerDataFile>,
) {
    let mut player_transform_lock = player_transform_mutex_handle.0.lock().unwrap();
    if *player_transform_lock != player_transform.translation {
        *player_transform_lock = player_transform.translation;
        write_player_position(&mut player_data_file.0, *player_transform_lock);
    }
}
