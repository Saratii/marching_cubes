use bevy::prelude::*;

use crate::player::player::{CameraController, KeyBindings};
use crate::ui::configurable_settings::ConfigurableSettings;

const LAMP_SIZE: f32 = 0.12; //edge length of the lamp lens cube
const LAMP_RANGE: f32 = 250.0; //hard distance cutoff of the spotlight
const LAMP_EMISSIVE: LinearRgba = LinearRgba::rgb(60_000.0, 57_000.0, 42_000.0);

#[derive(Component)]
pub struct HeadlampTag;

#[derive(Component)]
pub struct HeadlampLightTag;

pub fn spawn_headlamp(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    root: Entity,
    lamp_y: f32,
    front_z: f32,
    settings: &ConfigurableSettings,
) {
    let lamp_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.95, 0.8),
        emissive: LAMP_EMISSIVE,
        ..default()
    });
    let lamp_mesh = meshes.add(Cuboid::new(LAMP_SIZE, LAMP_SIZE, LAMP_SIZE));
    let (outer_angle, inner_angle) = settings.headlamp_angles();
    let lamp_z = front_z - LAMP_SIZE / 2.0;
    let lamp = commands
        .spawn((
            Mesh3d(lamp_mesh),
            MeshMaterial3d(lamp_material),
            HeadlampTag,
            Transform::from_xyz(0.0, lamp_y, lamp_z),
        ))
        .with_children(|lamp| {
            //Visibility::Visible keeps the light lit under the hidden mesh root
            lamp.spawn((
                SpotLight {
                    color: Color::srgb(1.0, 0.95, 0.8),
                    intensity: settings.headlamp_brightness,
                    range: LAMP_RANGE,
                    outer_angle,
                    inner_angle,
                    shadow_maps_enabled: true,
                    ..default()
                },
                HeadlampLightTag,
                Visibility::Visible,
                Transform::from_xyz(0.0, 0.0, -LAMP_SIZE),
            ));
        })
        .id();
    commands.entity(root).add_child(lamp);
}

pub fn aim_headlamp(
    camera_controller: Res<CameraController>,
    mut lamp_query: Query<&mut Transform, With<HeadlampTag>>,
) {
    let Ok(mut transform) = lamp_query.single_mut() else {
        return;
    };
    transform.rotation = Quat::from_rotation_x(camera_controller.player_pitch);
}

pub fn toggle_headlamp(
    keyboard: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    mut light_query: Query<&mut Visibility, With<HeadlampLightTag>>,
    lens_query: Query<&MeshMaterial3d<StandardMaterial>, With<HeadlampTag>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !keyboard.just_pressed(key_bindings.toggle_headlamp) {
        return;
    }
    let Ok(mut visibility) = light_query.single_mut() else {
        return;
    };
    let was_on = *visibility == Visibility::Visible;
    *visibility = if was_on {
        Visibility::Hidden
    } else {
        Visibility::Visible
    };
    let Ok(lens_material) = lens_query.single() else {
        return;
    };
    if let Some(mut material) = materials.get_mut(&lens_material.0) {
        material.emissive = if was_on {
            LinearRgba::BLACK
        } else {
            LAMP_EMISSIVE
        };
    }
}
