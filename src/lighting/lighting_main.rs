use std::f32::consts::FRAC_PI_4;

use bevy::{
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{AtmosphereEnvironmentMapLight, FogVolume, VolumetricFog, VolumetricLight},
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
};

use crate::player::player::{CAMERA_FIRST_PERSON_OFFSET, MainCameraTag};

pub fn setup_lighting(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 20000.,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 1.0, -FRAC_PI_4)),
        VolumetricLight,
    ));
}

pub fn setup_camera(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    commands
        .spawn((
            Camera3d::default(),
            Transform {
                translation: CAMERA_FIRST_PERSON_OFFSET,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            Camera::default(),
            IsDefaultUiCamera,
            MainCameraTag,
            Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
            AtmosphereSettings::default(),
            Exposure { ev100: 13.0 },
            Tonemapping::AcesFitted,
            Bloom::NATURAL,
            AtmosphereEnvironmentMapLight::default(),
            VolumetricFog {
                ambient_intensity: 0.0,
                ..default()
            },
            Msaa::Off,
            ScreenSpaceReflections::default(),
        ))
        .with_child((
            FogVolume {
                density_factor: 0.1,
                ..default()
            },
            Transform::from_scale(Vec3::new(30.0, 30.0, 30.0)).with_translation(Vec3::ZERO),
        ));
}
