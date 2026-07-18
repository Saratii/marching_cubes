use std::f32::consts::FRAC_PI_4;

use bevy::{
    camera::Exposure,
    core_pipeline::{prepass::DepthPrepass, tonemapping::Tonemapping},
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::{AtmosphereSettings, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
    render::occlusion_culling::OcclusionCulling,
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
            shadow_maps_enabled: true,
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
    mut commands: Commands,
    camera_entity_query: Query<Entity, With<MainCameraTag>>,
) {
    if !settings.is_changed() {
        return;
    }
    if let Ok(mut light) = light_query.single_mut() {
        light.shadow_maps_enabled = settings.shadows;
    }
    if let Ok(entity) = camera_entity_query.single() {
        if settings.distance_fog {
            let render_radius = settings.render_radius_squared.0.sqrt();
            if let Ok(mut fog) = fog_query.single_mut() {
                fog.falloff = FogFalloff::Linear {
                    start: render_radius * settings.fog_start_multiplier,
                    end: render_radius * settings.fog_end_multiplier,
                };
            } else {
                commands.entity(entity).insert(DistanceFog {
                    color: Color::srgb(0.8, 0.8, 0.9),
                    falloff: FogFalloff::Linear {
                        start: render_radius * settings.fog_start_multiplier,
                        end: render_radius * settings.fog_end_multiplier,
                    },
                    ..default()
                });
            }
        } else {
            commands.entity(entity).remove::<DistanceFog>();
        }
        if settings.occlusion_culling {
            commands
                .entity(entity)
                .insert((DepthPrepass, OcclusionCulling));
        } else {
            commands
                .entity(entity)
                .remove::<(DepthPrepass, OcclusionCulling)>();
        }
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
        OcclusionCulling,
        MainCameraTag,
        DepthPrepass,
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
    commands.spawn(Atmosphere {
        inner_radius: 6_360_000.0,
        outer_radius: 6_460_000.0,
        ground_albedo: Vec3::splat(0.3),
        medium: scattering_mediums.add(ScatteringMedium::default()),
    });
}
