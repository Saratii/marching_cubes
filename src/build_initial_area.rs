use bevy::prelude::*;

use crate::deformable_terrain::{
    digging::{chunk_coords_in_cylinder, chunk_coords_in_sphere},
    driver::TerrainChunkMap,
    plugin::{Deformation, TerrainHeightSource},
};

/// Radius of the spherical underground room.
const ROOM_RADIUS: f32 = 10.0;
/// Depth of the room's center below the terrain surface at the origin.
const ROOM_DEPTH: f32 = 25.0;
/// Radius of the vertical shaft connecting the surface to the room.
const SHAFT_RADIUS: f32 = 3.0;
/// How far above the surface the shaft carve starts, so it cleanly breaks
/// through the ground instead of leaving a thin skin over the opening.
const SHAFT_TOP_MARGIN: f32 = 2.0;

pub fn build_initial_area(
    terrain_chunk_map: Res<TerrainChunkMap>,
    height_source: Res<TerrainHeightSource>,
    mut deformation_writer: MessageWriter<Deformation>,
    mut dispatched: Local<bool>,
) {
    if *dispatched {
        return;
    }
    let surface_height = height_source.0.height_at(0.0, 0.0);
    let room_center = Vec3::new(0.0, surface_height - ROOM_DEPTH, 0.0);
    let shaft_top = surface_height + SHAFT_TOP_MARGIN;
    let shaft_half_height = (shaft_top - room_center.y) / 2.0;
    let shaft_center = Vec3::new(0.0, shaft_top - shaft_half_height, 0.0);
    let ready = {
        let map = terrain_chunk_map.0.lock().unwrap();
        chunk_coords_in_sphere(room_center, ROOM_RADIUS)
            .chain(chunk_coords_in_cylinder(
                shaft_center,
                SHAFT_RADIUS,
                shaft_half_height,
            ))
            .all(|coord| map.contains_key(&coord))
    };
    if !ready {
        return;
    }
    deformation_writer.write(Deformation::SphereCarve {
        center: room_center,
        radius: ROOM_RADIUS,
    });
    deformation_writer.write(Deformation::CylinderCarve {
        center: shaft_center,
        radius: SHAFT_RADIUS,
        half_height: shaft_half_height,
    });
    *dispatched = true;
}
