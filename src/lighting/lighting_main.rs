use std::f32::consts::FRAC_PI_4;

use bevy::{
    camera::Exposure,
    core_pipeline::{prepass::DepthPrepass, tonemapping::Tonemapping},
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, VolumetricFog, atmosphere::ScatteringMedium,
    },
    pbr::AtmosphereSettings,
    post_process::bloom::Bloom,
    prelude::*,
    render::occlusion_culling::OcclusionCulling,
};

use crate::{
    constants::CAMERA_FIRST_PERSON_OFFSET,
    elevator::GodRayLightTag,
    lanterns::LanternLightTag,
    player::{headlamp::HeadlampLightTag, player::MainCameraTag},
    ui::configurable_settings::ConfigurableSettings,
};

#[derive(Component)]
pub struct SunLightTag;

pub fn setup_lighting(mut commands: Commands, settings: Res<ConfigurableSettings>) {
    commands.spawn((
        DirectionalLight {
            illuminance: settings.sun_illuminance,
            shadow_maps_enabled: settings.shadows,
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
    mut ambient_query: Query<&mut AmbientLight, With<MainCameraTag>>,
    mut lantern_query: Query<&mut PointLight, With<LanternLightTag>>,
    mut exposure_query: Query<&mut Exposure, With<MainCameraTag>>,
    mut spot_light_query: Query<
        (&mut SpotLight, Has<GodRayLightTag>),
        Or<(With<GodRayLightTag>, With<HeadlampLightTag>)>,
    >,
) {
    if !settings.is_changed() {
        return;
    }
    if let Ok(mut light) = light_query.single_mut() {
        light.shadow_maps_enabled = settings.shadows;
        light.illuminance = settings.sun_illuminance;
    }
    for mut lantern in lantern_query.iter_mut() {
        lantern.intensity = settings.lantern_brightness;
    }
    for (mut spot_light, is_god_ray) in spot_light_query.iter_mut() {
        if is_god_ray {
            spot_light.intensity = settings.god_ray_brightness;
        } else {
            let (outer, inner) = settings.headlamp_angles();
            spot_light.intensity = settings.headlamp_brightness;
            spot_light.outer_angle = outer;
            spot_light.inner_angle = inner;
        }
    }
    if let Ok(mut ambient) = ambient_query.single_mut() {
        ambient.brightness = settings.ambient_brightness;
    }
    if let Ok(mut exposure) = exposure_query.single_mut() {
        exposure.ev100 = settings.exposure_ev100;
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
        Exposure {
            ev100: settings.exposure_ev100,
        },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        AmbientLight {
            brightness: settings.ambient_brightness,
            ..default()
        },
        Msaa::Off,
        //ambient 0 so fog volumes only glow where a VolumetricLight hits them
        VolumetricFog {
            ambient_intensity: 0.0,
            ..default()
        },
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
