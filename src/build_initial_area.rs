use bevy::prelude::*;

use crate::deformable_terrain::{
    digging::chunk_coords_in_cylinder,
    driver::TerrainChunkMap,
    plugin::{Deformation, TerrainHeightSource},
};

/// Radius of the cylindrical underground room (flat floor and flat ceiling).
pub const ROOM_RADIUS: f32 = 15.0;
/// Height of the room, floor to ceiling.
pub const ROOM_HEIGHT: f32 = 15.0;
/// Depth of the room's flat floor below the terrain surface at the origin.
pub const ROOM_DEPTH: f32 = 45.0;
/// Radius of the vertical shaft connecting the surface to the room, as the
/// elevator is sized: platforms are this wide, and the shaft liner wraps just
/// outside it.
pub const SHAFT_RADIUS: f32 = 3.0;
/// Extra radius on the shaft's dirt carve beyond `SHAFT_RADIUS`: the liner
/// tube around the shaft spans 0.05 (platform clearance) to 0.35 (wall
/// thickness) outside the platforms, so the dirt must recede past 0.35 or the
/// liner ends up buried behind it; the last 0.05 is an air gap hidden behind
/// the liner wall.
const SHAFT_CARVE_MARGIN: f32 = 0.4;
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
    let floor_y = surface_height - ROOM_DEPTH;
    let room_center = Vec3::new(0.0, floor_y + ROOM_HEIGHT / 2.0, 0.0);
    let shaft_top = surface_height + SHAFT_TOP_MARGIN;
    let shaft_half_height = (shaft_top - floor_y) / 2.0;
    let shaft_center = Vec3::new(0.0, shaft_top - shaft_half_height, 0.0);
    let ready = {
        let map = terrain_chunk_map.0.lock().unwrap();
        chunk_coords_in_cylinder(room_center, ROOM_RADIUS, ROOM_HEIGHT / 2.0, Quat::IDENTITY)
            .chain(chunk_coords_in_cylinder(
                shaft_center,
                SHAFT_RADIUS + SHAFT_CARVE_MARGIN,
                shaft_half_height,
                Quat::IDENTITY,
            ))
            .all(|coord| map.contains_key(&coord))
    };
    if !ready {
        return;
    }
    deformation_writer.write(Deformation::CylinderCarve {
        center: room_center,
        radius: ROOM_RADIUS,
        half_height: ROOM_HEIGHT / 2.0,
        rotation: Quat::IDENTITY,
    });
    deformation_writer.write(Deformation::CylinderCarve {
        center: shaft_center,
        radius: SHAFT_RADIUS + SHAFT_CARVE_MARGIN,
        half_height: shaft_half_height,
        rotation: Quat::IDENTITY,
    });
    *dispatched = true;
}
