use std::f32::consts::FRAC_PI_4;

use bevy::{
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::AtmosphereEnvironmentMapLight,
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
};

use crate::{
    constants::CAMERA_FIRST_PERSON_OFFSET, player::player::MainCameraTag,
    ui::configurable_settings::ConfigurableSettings,
};

#[derive(Component)]
pub struct SunLightTag;

pub fn setup_lighting(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 80000.,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 1.0, -FRAC_PI_4)),
        SunLightTag,
    ));
}

pub fn apply_settings_changes(
    settings: Res<ConfigurableSettings>,
    mut light_query: Query<&mut DirectionalLight, With<SunLightTag>>,
    mut fog_query: Query<&mut DistanceFog, With<MainCameraTag>>,
) {
    if !settings.is_changed() {
        return;
    }
    if let Ok(mut light) = light_query.single_mut() {
        light.shadows_enabled = settings.shadows;
    }
    if let Ok(mut fog) = fog_query.single_mut() {
        let render_radius = settings.render_radius_squared.0.sqrt();
        fog.falloff = FogFalloff::Linear {
            start: render_radius * settings.fog_start_multiplier,
            end: render_radius * settings.fog_end_multiplier,
        };
    }
}

pub fn setup_camera(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    settings: Res<ConfigurableSettings>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)));
    let render_radius = settings.render_radius_squared.0.sqrt();
    commands.spawn((
        Camera3d::default(),
        Transform {
            translation: CAMERA_FIRST_PERSON_OFFSET,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        Camera::default(),
        IsDefaultUiCamera,
        MainCameraTag,
        Atmosphere {
            bottom_radius: 6_360_000.0,
            top_radius: 6_460_000.0,
            ground_albedo: Vec3::splat(0.3),
            medium: scattering_mediums.add(ScatteringMedium::default()),
        },
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        Msaa::Off,
        ScreenSpaceReflections::default(),
        DistanceFog {
            color: Color::srgb(0.8, 0.8, 0.9),
            falloff: FogFalloff::Linear {
                start: render_radius * settings.fog_start_multiplier,
                end: render_radius * settings.fog_end_multiplier,
            },
            ..default()
        },
    ));
}
