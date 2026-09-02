use std::f32::consts::PI;

use bevy::prelude::*;

use crate::{
    constants::PLAYER_CUBOID_SIZE,
    player::{
        headlamp::spawn_headlamp,
        player::{PlayerMeshTag, PlayerTag},
    },
    ui::configurable_settings::ConfigurableSettings,
};

const HALF_HEIGHT: f32 = PLAYER_CUBOID_SIZE.y / 2.0;
const LEG_LENGTH: f32 = HALF_HEIGHT;
const LEG_WIDTH: f32 = 0.2;
const LEG_DEPTH: f32 = 0.4;
const LEG_OFFSET_X: f32 = 0.13;
const ARM_LENGTH: f32 = 0.65;
const ARM_THICKNESS: f32 = 0.16;
const ARM_OFFSET_X: f32 = PLAYER_CUBOID_SIZE.x / 2.0 + ARM_THICKNESS / 2.0;
const SHOULDER_HEIGHT: f32 = HALF_HEIGHT - ARM_THICKNESS / 2.0;
const HEAD_SIZE: f32 = 0.4;
const HEAD_CENTER_Y: f32 = HALF_HEIGHT + HEAD_SIZE / 2.0;
const HELMET_WIDTH: f32 = HEAD_SIZE + 0.08;
const HELMET_HEIGHT: f32 = 0.2;
const HELMET_BRIM_DEPTH: f32 = 0.14;
const SWING_SPEED: f32 = 8.0;
const LEG_SWING_AMPLITUDE: f32 = 0.7;
const ARM_SWING_AMPLITUDE: f32 = 0.5;
//horizontal speed above which the walk cycle plays
const MOVE_SPEED_THRESHOLD: f32 = 0.5;
const SWING_BLEND_SPEED: f32 = 6.0;

/// Where a held rock rides: just past the right fist, a touch in front of it
/// (the player mesh faces -Z).
#[derive(Component)]
pub struct PlayerHandTag;

#[derive(Component)]
pub struct SwingingLimb {
    pub phase_offset: f32,
    pub amplitude: f32,
}

//builds the player body: torso cube on the upper half, legs pivoting at the
//hips on the lower half, arms pivoting at the shoulders
pub fn spawn_player_visual(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    settings: &ConfigurableSettings,
) -> Entity {
    let material: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.3, 0.3),
        ..default()
    });
    let torso_mesh = meshes.add(Cuboid::new(
        PLAYER_CUBOID_SIZE.x,
        HALF_HEIGHT,
        PLAYER_CUBOID_SIZE.z,
    ));
    let leg_mesh = meshes.add(Cuboid::new(LEG_WIDTH, LEG_LENGTH, LEG_DEPTH));
    let arm_mesh = meshes.add(Cuboid::new(ARM_THICKNESS, ARM_LENGTH, ARM_THICKNESS));
    let root = commands
        .spawn((Transform::default(), Visibility::Hidden, PlayerMeshTag))
        .id();
    let torso = commands
        .spawn((
            Mesh3d(torso_mesh),
            MeshMaterial3d(material.clone()),
            Transform::from_xyz(0.0, HALF_HEIGHT / 2.0, 0.0),
        ))
        .id();
    commands.entity(root).add_child(torso);
    spawn_head_and_helmet(commands, meshes, materials, root, settings);
    //opposite legs and arms swing in opposite phase
    for (x, phase_offset) in [(-LEG_OFFSET_X, 0.0), (LEG_OFFSET_X, PI)] {
        let hip = commands
            .spawn((
                Transform::from_xyz(x, 0.0, 0.0),
                Visibility::default(),
                SwingingLimb {
                    phase_offset,
                    amplitude: LEG_SWING_AMPLITUDE,
                },
            ))
            .id();
        let leg = commands
            .spawn((
                Mesh3d(leg_mesh.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(0.0, -LEG_LENGTH / 2.0, 0.0),
            ))
            .id();
        commands.entity(hip).add_child(leg);
        commands.entity(root).add_child(hip);
    }
    for (x, phase_offset) in [(-ARM_OFFSET_X, PI), (ARM_OFFSET_X, 0.0)] {
        let shoulder = commands
            .spawn((
                Transform::from_xyz(x, SHOULDER_HEIGHT, 0.0),
                Visibility::default(),
                SwingingLimb {
                    phase_offset,
                    amplitude: ARM_SWING_AMPLITUDE,
                },
            ))
            .id();
        let arm = commands
            .spawn((
                Mesh3d(arm_mesh.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(0.0, -ARM_LENGTH / 2.0, 0.0),
            ))
            .id();
        if x > 0.0 {
            let hand = commands
                .spawn((
                    Transform::from_xyz(0.0, -ARM_LENGTH / 2.0, -ARM_THICKNESS),
                    Visibility::default(),
                    PlayerHandTag,
                ))
                .id();
            commands.entity(arm).add_child(hand);
        }
        commands.entity(shoulder).add_child(arm);
        commands.entity(root).add_child(shoulder);
    }
    root
}

//head cube topped by a mining helmet whose lamp casts a real spotlight;
//the player faces -Z, which is also the direction a Bevy spotlight shines
fn spawn_head_and_helmet(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    root: Entity,
    settings: &ConfigurableSettings,
) {
    let skin = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.68, 0.55),
        ..default()
    });
    let helmet_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.75, 0.1),
        metallic: 0.3,
        perceptual_roughness: 0.5,
        ..default()
    });
    let head_mesh = meshes.add(Cuboid::new(HEAD_SIZE, HEAD_SIZE, HEAD_SIZE));
    let dome_mesh = meshes.add(Cuboid::new(HELMET_WIDTH, HELMET_HEIGHT, HELMET_WIDTH));
    let brim_mesh = meshes.add(Cuboid::new(HELMET_WIDTH, 0.05, HELMET_BRIM_DEPTH));
    let head = commands
        .spawn((
            Mesh3d(head_mesh),
            MeshMaterial3d(skin),
            Transform::from_xyz(0.0, HEAD_CENTER_Y, 0.0),
        ))
        .id();
    commands.entity(root).add_child(head);
    let helmet_base_y = HEAD_CENTER_Y + HEAD_SIZE / 2.0;
    let dome = commands
        .spawn((
            Mesh3d(dome_mesh),
            MeshMaterial3d(helmet_material.clone()),
            Transform::from_xyz(0.0, helmet_base_y + HELMET_HEIGHT / 2.0, 0.0),
        ))
        .id();
    commands.entity(root).add_child(dome);
    let front_z = -(HEAD_SIZE / 2.0);
    let brim = commands
        .spawn((
            Mesh3d(brim_mesh),
            MeshMaterial3d(helmet_material),
            Transform::from_xyz(0.0, helmet_base_y, front_z - HELMET_BRIM_DEPTH / 2.0),
        ))
        .id();
    commands.entity(root).add_child(brim);
    let lamp_y = helmet_base_y + HELMET_HEIGHT / 2.0;
    spawn_headlamp(commands, meshes, materials, root, lamp_y, front_z, settings);
}

pub fn animate_player_limbs(
    time: Res<Time>,
    player_query: Query<&Transform, With<PlayerTag>>,
    mut limb_query: Query<(&SwingingLimb, &mut Transform), Without<PlayerTag>>,
    mut phase: Local<f32>,
    mut blend: Local<f32>,
    mut last_position: Local<Option<Vec3>>,
) {
    let Ok(player_transform) = player_query.single() else {
        return;
    };
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let position = player_transform.translation;
    let horizontal_speed = match *last_position {
        Some(last) => Vec2::new(position.x - last.x, position.z - last.z).length() / dt,
        None => 0.0,
    };
    *last_position = Some(position);
    let moving = horizontal_speed > MOVE_SPEED_THRESHOLD;
    let target = if moving { 1.0 } else { 0.0 };
    let max_step = SWING_BLEND_SPEED * dt;
    *blend += (target - *blend).clamp(-max_step, max_step);
    if moving {
        *phase += SWING_SPEED * dt;
    } else if *blend <= 0.0 {
        *phase = 0.0;
    }
    for (limb, mut transform) in limb_query.iter_mut() {
        let angle = (*phase + limb.phase_offset).sin() * limb.amplitude * *blend;
        transform.rotation = Quat::from_rotation_x(angle);
    }
}
