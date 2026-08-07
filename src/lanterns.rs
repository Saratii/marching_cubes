use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::build_initial_area::{ROOM_DEPTH, ROOM_HEIGHT};
use crate::deformable_terrain::plugin::TerrainHeightSource;

/// Number of rigid links in each hanging chain.
const LINK_COUNT: usize = 6;
/// Radius of the chain link capsule colliders.
const LINK_RADIUS: f32 = 0.05;
/// Gap between the cavern's roof plate and the chain anchor.
const ATTACH_INSET: f32 = 0.15;
/// Distance from the lantern's center up to the hook where the chain attaches.
const HOOK_OFFSET: f32 = 0.7;
/// Distance from the chain anchor down to the lantern's center at rest.
const LANTERN_DROP: f32 = 4.0;
/// Distance from the room's axis to each lantern.
const LANTERN_RING_RADIUS: f32 = 8.0;
/// Hanging spots, offset from the player spawn direction (angle 0) and the
/// support beams.
const LANTERN_ANGLES_DEGREES: [f32; 3] = [90.0, 210.0, 330.0];
/// Half extents of the lantern's box collider.
const LANTERN_HALF_EXTENTS: Vec3 = Vec3::new(0.3, 0.5, 0.3);

pub fn spawn_lanterns(
    mut commands: Commands,
    height_source: Res<TerrainHeightSource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut spawned: Local<bool>,
) {
    if *spawned {
        return;
    }
    *spawned = true;
    let floor_y = height_source.0.height_at(0.0, 0.0) - ROOM_DEPTH;
    let attach_y = floor_y + ROOM_HEIGHT - ATTACH_INSET;
    let chain_span = LANTERN_DROP - HOOK_OFFSET;
    let link_length = chain_span / LINK_COUNT as f32;
    //slightly overlong so link meshes overlap at the joints instead of gapping
    let link_mesh = meshes.add(Cylinder::new(LINK_RADIUS * 0.8, link_length * 1.05));
    let iron = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.22, 0.25),
        metallic: 0.9,
        perceptual_roughness: 0.5,
        ..default()
    });
    let cap_mesh = meshes.add(Cuboid::new(0.56, 0.2, 0.56));
    let base_mesh = meshes.add(Cuboid::new(0.44, 0.12, 0.44));
    let glow_mesh = meshes.add(Sphere::new(0.24));
    let glow_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.5),
        emissive: LinearRgba::rgb(1.0, 0.55, 0.22) * 40_000.0,
        ..default()
    });
    for angle in LANTERN_ANGLES_DEGREES.map(f32::to_radians) {
        let x = LANTERN_RING_RADIUS * angle.cos();
        let z = LANTERN_RING_RADIUS * angle.sin();
        let mut parent = commands
            .spawn((RigidBody::Fixed, Transform::from_xyz(x, attach_y, z)))
            .id();
        let mut parent_anchor = Vec3::ZERO;
        for i in 0..LINK_COUNT {
            let joint = SphericalJointBuilder::new()
                .local_anchor1(parent_anchor)
                .local_anchor2(Vec3::new(0.0, link_length / 2.0, 0.0));
            parent = commands
                .spawn((
                    RigidBody::Dynamic,
                    Collider::capsule_y(link_length / 2.0 - LINK_RADIUS, LINK_RADIUS),
                    ColliderMassProperties::Mass(0.8),
                    Damping {
                        linear_damping: 0.4,
                        angular_damping: 0.6,
                    },
                    Mesh3d(link_mesh.clone()),
                    MeshMaterial3d(iron.clone()),
                    Transform::from_xyz(x, attach_y - link_length * (i as f32 + 0.5), z),
                    ImpulseJoint::new(parent, joint),
                ))
                .id();
            parent_anchor = Vec3::new(0.0, -link_length / 2.0, 0.0);
        }
        let joint = SphericalJointBuilder::new()
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(0.0, HOOK_OFFSET, 0.0));
        commands
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(
                    LANTERN_HALF_EXTENTS.x,
                    LANTERN_HALF_EXTENTS.y,
                    LANTERN_HALF_EXTENTS.z,
                ),
                ColliderMassProperties::Mass(6.0),
                Damping {
                    linear_damping: 0.3,
                    angular_damping: 0.5,
                },
                Transform::from_xyz(x, attach_y - LANTERN_DROP, z),
                Visibility::default(),
                ImpulseJoint::new(parent, joint),
            ))
            .with_children(|children| {
                children.spawn((
                    Mesh3d(cap_mesh.clone()),
                    MeshMaterial3d(iron.clone()),
                    Transform::from_xyz(0.0, LANTERN_HALF_EXTENTS.y - 0.1, 0.0),
                ));
                children.spawn((
                    Mesh3d(base_mesh.clone()),
                    MeshMaterial3d(iron.clone()),
                    Transform::from_xyz(0.0, -LANTERN_HALF_EXTENTS.y + 0.06, 0.0),
                ));
                children.spawn((
                    Mesh3d(glow_mesh.clone()),
                    MeshMaterial3d(glow_material.clone()),
                    Transform::from_xyz(0.0, -0.04, 0.0),
                ));
                children.spawn((
                    PointLight {
                        color: Color::srgb(1.0, 0.72, 0.45),
                        intensity: 6_000_000.0,
                        range: 30.0,
                        ..default()
                    },
                    Transform::default(),
                ));
            });
    }
}
