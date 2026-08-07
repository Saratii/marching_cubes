use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::build_initial_area::{ROOM_DEPTH, ROOM_RADIUS, SHAFT_RADIUS};
use crate::constants::PLAYER_CUBOID_SIZE;
use crate::deformable_terrain::plugin::TerrainHeightSource;
use crate::player::player::PlayerTag;

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
/// Vertical thickness of the flat base discs at each end of the shaft liner.
const BASE_THICKNESS: f32 = 0.15;
/// How far beyond the liner wall the flat base discs extend.
const BASE_EXTRA_RADIUS: f32 = 0.4;
/// Extra drop of the bottom base disc below the liner's end, so its outer rim
/// clears the dome ceiling (which curves down as it moves away from the shaft).
const BOTTOM_BASE_DROP: f32 = 0.6;

#[derive(Component)]
pub struct ElevatorPlatform;

#[derive(Resource)]
pub struct Elevator {
    bottom_y: f32,
    spawn_timer: Timer,
    platform_mesh: Handle<Mesh>,
    platform_material: Handle<StandardMaterial>,
}

pub fn setup_elevator(
    mut commands: Commands,
    height_source: Res<TerrainHeightSource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let surface_y = height_source.0.height_at(0.0, 0.0);
    let bottom_y = surface_y - ROOM_DEPTH;
    let silver = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.4),
        metallic: 0.8,
        perceptual_roughness: 0.4,
        ..default()
    });
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
    //liner reinforcing the upper shaft: starts where the shaft pierces the room's
    //dome (so the room itself stays open) and runs up to the surface. Extended
    //down by the base drop so it stays attached to the lowered bottom disc.
    let dome_pierce_y = bottom_y + (ROOM_RADIUS * ROOM_RADIUS - SHAFT_RADIUS * SHAFT_RADIUS).sqrt();
    let liner_bottom = dome_pierce_y - BOTTOM_BASE_DROP;
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
    //flat base discs hiding the seam between the liner and the dirt: one flush
    //under the liner's bottom at the room's dome, one resting on the ground
    //around the collar at the surface. Both reach the same outer radius; the top
    //one starts at the narrower collar wall so there's no gap between them.
    let base_outer_radius = TUBE_INNER_RADIUS + RING_THICKNESS + BASE_EXTRA_RADIUS;
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver.clone(),
        liner_bottom + BASE_THICKNESS / 2.0,
        BASE_THICKNESS,
        TUBE_INNER_RADIUS,
        base_outer_radius - TUBE_INNER_RADIUS,
    );
    spawn_shaft_tube(
        &mut commands,
        &mut meshes,
        silver,
        surface_y + BASE_THICKNESS / 2.0,
        BASE_THICKNESS,
        COLLAR_INNER_RADIUS,
        base_outer_radius - COLLAR_INNER_RADIUS,
    );
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
            ElevatorPlatform,
        ));
    }
}
