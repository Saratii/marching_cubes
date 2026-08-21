use std::f32::consts::{FRAC_PI_2, TAU};

use bevy::asset::RenderAssetUsages;
use bevy::light::{FogVolume, NotShadowCaster, VolumetricLight};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_rapier3d::prelude::*;

use crate::build_initial_area::{ROOM_DEPTH, ROOM_HEIGHT, ROOM_RADIUS, SHAFT_RADIUS};
use crate::constants::PLAYER_CUBOID_SIZE;
use crate::deformable_terrain::plugin::TerrainHeightSource;
use crate::player::player::PlayerTag;
use crate::ui::configurable_settings::ConfigurableSettings;

/// Y position at which a platform despawns.
const ELEVATOR_TOP_Y: f32 = 10.0;
/// Seconds for one platform to travel from the bottom of the hole to the top.
const TRAVEL_SECONDS: f32 = 20.0;
/// Seconds between platform spawns, escalator style.
const SPAWN_INTERVAL_SECONDS: f32 = 10.0;
/// Thickness of each platform disc.
const PLATFORM_THICKNESS: f32 = 0.2;
/// Height of the static ring that hides platforms popping in at the bottom.
const RING_HEIGHT: f32 = 0.6;
/// Radial thickness of the ring wall.
const RING_THICKNESS: f32 = 0.3;
/// Gap between the platform edge and the ring's inner wall.
const RING_CLEARANCE: f32 = 0.05;
/// Inner radius of the stationary tubes, leaving clearance around the platforms.
const TUBE_INNER_RADIUS: f32 = SHAFT_RADIUS + RING_CLEARANCE;
/// Radius of the moving platform discs: same as the initial dig.
const PLATFORM_RADIUS: f32 = SHAFT_RADIUS;
/// Vertical distance between the player's feet and a platform top within which
/// the player counts as standing on it and gets carried along.
const RIDE_MARGIN: f32 = 0.4;
/// How far the collar sticks up above the terrain surface.
const COLLAR_HEIGHT: f32 = 0.8;
/// Wall thickness of the above-ground collar; thinner than the liner so it
/// extends less far out around the shaft mouth.
const COLLAR_THICKNESS: f32 = 0.15;
/// Inner radius of the collar; narrower than the shaft.
const COLLAR_INNER_RADIUS: f32 = SHAFT_RADIUS - 0.5;
/// Vertical thickness of the flat base disc resting on the ground around the
/// collar at the surface.
const BASE_THICKNESS: f32 = 0.15;
/// How far beyond the liner wall the flat base disc extends.
const BASE_EXTRA_RADIUS: f32 = 0.4;
/// Vertical thickness of the flat plates lining the cavern's floor and
/// ceiling. Kept at most the player's autostep height so the floor plate's
/// edge is walkable.
const PLATE_THICKNESS: f32 = 0.1;
/// Outer radius of the floor and roof plates: extends past the room radius so
/// the plates' rims bury into the cavern's wall, hiding the dirt seams.
const PLATE_OUTER_RADIUS: f32 = ROOM_RADIUS + 0.5;
/// Number of support beams holding the roof plate up.
const BEAM_COUNT: usize = 6;
/// Radius of each support beam.
const BEAM_RADIUS: f32 = 0.4;
/// Distance from the room's axis to each beam's axis, placing the ring of
/// beams just inside the cavern wall.
const BEAM_RING_RADIUS: f32 = ROOM_RADIUS - 1.5;
/// Height of the god-ray spotlight above the terrain surface. Higher makes the
/// beam more parallel (thinner cone) but needs more range and intensity.
const GOD_RAY_LIGHT_HEIGHT: f32 = 20.0;
/// Range of the god-ray spotlight. Must be far beyond the cavern floor:
/// attenuation ramps down as (1 - (d/range)^4)^2, so a range just past the
/// floor leaves almost no light there.
const GOD_RAY_RANGE: f32 = 200.0;
/// Horizontal extent of the fog volume box wrapping the beam. Kept snug around
/// the shaft so raymarching cost stays low and no other lights catch the fog.
const GOD_RAY_FOG_WIDTH: f32 = 12.0;

#[derive(Component)]
pub struct ElevatorPlatform;

#[derive(Component)]
pub struct GodRayLightTag;

#[derive(Resource)]
pub struct Elevator {
    bottom_y: f32,
    spawn_timer: Timer,
    platform_mesh: Handle<Mesh>,
    platform_material: Handle<StandardMaterial>,
}

/// 3D density texture for the god-ray fog: full density around the beam,
/// smoothly fading to zero before the volume's walls so the box's rectangular
/// bounds never show as a visible seam against the background.
fn god_ray_density_image() -> Image {
    const N: usize = 32;
    let mut data = vec![0u8; N * N * N];
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let dx = x as f32 / (N - 1) as f32 * 2.0 - 1.0;
                let dz = z as f32 / (N - 1) as f32 * 2.0 - 1.0;
                let r = (dx * dx + dz * dz).sqrt();
                let t = ((1.0 - r) / 0.3).clamp(0.0, 1.0);
                let d = t * t * (3.0 - 2.0 * t);
                data[(z * N + y) * N + x] = (d * 255.0) as u8;
            }
        }
    }
    Image::new(
        Extent3d {
            width: N as u32,
            height: N as u32,
            depth_or_array_layers: N as u32,
        },
        TextureDimension::D3,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

pub fn setup_elevator(
    mut commands: Commands,
    height_source: Res<TerrainHeightSource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    settings: Res<ConfigurableSettings>,
) {
    let surface_y = height_source.0.height_at(0.0, 0.0);
    let bottom_y = surface_y - ROOM_DEPTH;
    let silver = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.4),
        metallic: 0.8,
        perceptual_roughness: 0.4,
        ..default()
    });
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        bottom_y + PLATE_THICKNESS / 2.0,
        PLATE_THICKNESS,
        TUBE_INNER_RADIUS,
        PLATE_OUTER_RADIUS - TUBE_INNER_RADIUS,
    );
    //short ring at the room floor hiding where platforms pop in
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        bottom_y + RING_HEIGHT / 2.0,
        RING_HEIGHT,
        TUBE_INNER_RADIUS,
        RING_THICKNESS,
    );
    let ceiling_y = bottom_y + ROOM_HEIGHT;
    let liner_bottom = ceiling_y - PLATE_THICKNESS;
    let liner_height = surface_y - liner_bottom;
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        liner_bottom + liner_height / 2.0,
        liner_height,
        TUBE_INNER_RADIUS,
        RING_THICKNESS,
    );
    //narrow thin-walled collar poking up out of the shaft mouth
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        surface_y + COLLAR_HEIGHT / 2.0,
        COLLAR_HEIGHT,
        COLLAR_INNER_RADIUS,
        COLLAR_THICKNESS,
    );
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        ceiling_y - PLATE_THICKNESS / 2.0,
        PLATE_THICKNESS,
        TUBE_INNER_RADIUS,
        PLATE_OUTER_RADIUS - TUBE_INNER_RADIUS,
    );
    let beam_mesh = meshes.add(Cylinder::new(BEAM_RADIUS, ROOM_HEIGHT));
    for i in 0..BEAM_COUNT {
        let angle = i as f32 / BEAM_COUNT as f32 * TAU;
        commands.spawn((
            Mesh3d(beam_mesh.clone()),
            MeshMaterial3d(silver.clone()),
            Collider::cylinder(ROOM_HEIGHT / 2.0, BEAM_RADIUS),
            Transform::from_translation(Vec3::new(
                BEAM_RING_RADIUS * angle.cos(),
                bottom_y + ROOM_HEIGHT / 2.0,
                BEAM_RING_RADIUS * angle.sin(),
            )),
        ));
    }
    let base_outer_radius = TUBE_INNER_RADIUS + RING_THICKNESS + BASE_EXTRA_RADIUS;
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver,
        surface_y + BASE_THICKNESS / 2.0,
        BASE_THICKNESS,
        COLLAR_INNER_RADIUS,
        base_outer_radius - COLLAR_INNER_RADIUS,
    );
    //god ray: a thin white volumetric beam shining down the shaft. The cone is
    //sized to enter through the collar mouth; the liner walls' shadows clip it
    //to the shaft on the way down.
    let outer_angle = (COLLAR_INNER_RADIUS / GOD_RAY_LIGHT_HEIGHT).atan();
    commands.spawn((
        SpotLight {
            color: Color::WHITE,
            intensity: settings.god_ray_brightness,
            range: GOD_RAY_RANGE,
            shadow_maps_enabled: true,
            inner_angle: outer_angle * 0.7,
            outer_angle,
            ..default()
        },
        VolumetricLight,
        GodRayLightTag,
        Transform::from_xyz(0.0, surface_y + GOD_RAY_LIGHT_HEIGHT, 0.0)
            .looking_at(Vec3::new(0.0, bottom_y, 0.0), Vec3::Z),
    ));
    let fog_top = surface_y + COLLAR_HEIGHT;
    commands.spawn((
        //thin fog: extinction is exp(-density*(absorption+scattering)*meters),
        //so over the ~50m beam the density must stay low or the lower half of
        //the beam is invisible. light_intensity compensates, fog-only.
        FogVolume {
            density_factor: 0.04,
            absorption: 0.05,
            scattering: 0.5,
            light_intensity: 8.0,
            density_texture: Some(images.add(god_ray_density_image())),
            ..default()
        },
        Transform::from_xyz(0.0, (bottom_y + fog_top) / 2.0, 0.0).with_scale(Vec3::new(
            GOD_RAY_FOG_WIDTH,
            fog_top - bottom_y,
            GOD_RAY_FOG_WIDTH,
        )),
    ));
    commands.insert_resource(Elevator {
        bottom_y,
        spawn_timer: Timer::from_seconds(SPAWN_INTERVAL_SECONDS, TimerMode::Repeating),
        platform_mesh: meshes.add(Cylinder::new(PLATFORM_RADIUS, PLATFORM_THICKNESS)),
        platform_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.8, 0.85),
            metallic: 0.9,
            perceptual_roughness: 0.3,
            ..default()
        }),
    });
}

/// Spawns a stationary hollow cylinder (with trimesh collider) around the shaft
/// axis. Thin-and-tall makes a tube, wide-and-flat makes a base disc.
fn spawn_shaft_tube(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
    center_y: f32,
    height: f32,
    inner_radius: f32,
    wall_thickness: f32,
) {
    let mesh = Mesh::from(Extrusion::new(
        Annulus::new(inner_radius, inner_radius + wall_thickness),
        height,
    ));
    let collider = Collider::from_bevy_mesh(
        &mesh,
        &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
    )
    .expect("shaft tube trimesh collider failed to build");
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        collider,
        MeshMaterial3d(material),
        //extrusions extend along z, so tip the tube upright
        Transform::from_translation(Vec3::new(0.0, center_y, 0.0))
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    ));
}

pub fn update_elevator(
    mut commands: Commands,
    time: Res<Time>,
    mut elevator: ResMut<Elevator>,
    mut platforms: Query<(Entity, &mut Transform), With<ElevatorPlatform>>,
    mut player_query: Query<&mut Transform, (With<PlayerTag>, Without<ElevatorPlatform>)>,
) {
    let delta_y = (ELEVATOR_TOP_Y - elevator.bottom_y) / TRAVEL_SECONDS * time.delta_secs();
    //the character controller only resolves collisions when the player moves, so a platform
    //rising into the player interpenetrates and then gets ignored, dropping the player through.
    //Lift the rider by the platform's delta before moving the platform so they never overlap.
    if let Ok(mut player_transform) = player_query.single_mut() {
        let feet_y = player_transform.translation.y - PLAYER_CUBOID_SIZE.y / 2.0;
        let on_axis = player_transform.translation.xz().length()
            <= PLATFORM_RADIUS + PLAYER_CUBOID_SIZE.x / 2.0;
        let riding = on_axis
            && platforms.iter().any(|(_, platform)| {
                let platform_top = platform.translation.y + PLATFORM_THICKNESS / 2.0;
                (feet_y - platform_top).abs() <= RIDE_MARGIN
            });
        if riding {
            player_transform.translation.y += delta_y;
        }
    }
    for (entity, mut transform) in platforms.iter_mut() {
        transform.translation.y += delta_y;
        if transform.translation.y >= ELEVATOR_TOP_Y {
            commands.entity(entity).despawn();
        }
    }
    if elevator.spawn_timer.tick(time.delta()).just_finished() {
        commands.spawn((
            Mesh3d(elevator.platform_mesh.clone()),
            MeshMaterial3d(elevator.platform_material.clone()),
            Transform::from_translation(Vec3::new(0.0, elevator.bottom_y, 0.0)),
            Collider::cylinder(PLATFORM_THICKNESS / 2.0, PLATFORM_RADIUS),
            RigidBody::KinematicPositionBased,
            //platforms would otherwise shadow out the god ray while rising
            NotShadowCaster,
            ElevatorPlatform,
        ));
    }
}
