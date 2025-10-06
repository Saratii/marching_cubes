use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::camera::Viewport,
    window::{CursorGrabMode, PrimaryWindow},
};
use bevy_rapier3d::prelude::*;

const CAMERA_3RD_PERSON_OFFSET: Vec3 = Vec3 {
    x: 0.0,
    y: 5.0,
    z: 10.0,
};
const PLAYER_SPEED: f32 = 15.0;
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0., 10., 0.);
const PLAYER_CUBOID_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);
const CAMERA_FIRST_PERSON_OFFSET: Vec3 = Vec3::new(0., 0.75 * PLAYER_CUBOID_SIZE.y, 0.);
const MIN_ZOOM_DISTANCE: f32 = 4.0;
const MAX_ZOOM_DISTANCE: f32 = 100.0;
const ZOOM_SPEED: f32 = 5.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
const MIN_PITCH: f32 = -1.5;
const MAX_PITCH: f32 = 1.5;
const GRAVITY: f32 = -9.81;
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
    camera_controller: Res<CameraController>,
) {
    const MINIMAP_SIZE: u32 = 200;
    const BORDER_WIDTH: u32 = 3;
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
            Camera { ..default() },
            IsDefaultUiCamera,
            MainCameraTag,
        ))
        .with_child((
            Camera3d { ..default() },
            Transform::from_translation(Vec3::new(0., 150., 0.))
                .looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
            Camera {
                order: 1,
                viewport: Some(Viewport {
                    physical_position: UVec2::new(5, 5),
                    physical_size: UVec2::new(MINIMAP_SIZE, MINIMAP_SIZE),
                    ..default()
                }),
                ..default()
            },
        ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(5.0),
            top: Val::Px(5.0),
            width: Val::Px((MINIMAP_SIZE + BORDER_WIDTH * 2) as f32),
            height: Val::Px((MINIMAP_SIZE + BORDER_WIDTH * 2) as f32),
            border: UiRect::all(Val::Px(BORDER_WIDTH as f32)),
            ..default()
        })
        .insert(BorderColor(Color::BLACK))
        .insert(BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.5)));
}

pub fn toggle_camera(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_transform: Single<&mut Transform, With<MainCameraTag>>,
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
    mut camera_transform: Single<&mut Transform, With<MainCameraTag>>,
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
