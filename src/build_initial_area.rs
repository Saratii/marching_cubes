use bevy::prelude::*;

use crate::deformable_terrain::{
    digging::chunk_coords_in_sphere, driver::TerrainChunkMap, plugin::Deformation,
};

const INITIAL_AREA_CENTER: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const INITIAL_AREA_RADIUS: f32 = 10.0;

pub fn build_initial_area(
    terrain_chunk_map: Res<TerrainChunkMap>,
    mut deformation_writer: MessageWriter<Deformation>,
    mut dispatched: Local<bool>,
) {
    if *dispatched {
        return;
    }
    let ready = {
        let map = terrain_chunk_map.0.lock().unwrap();
        chunk_coords_in_sphere(INITIAL_AREA_CENTER, INITIAL_AREA_RADIUS)
            .all(|coord| map.contains_key(&coord))
    };
    if !ready {
        return;
    }
    deformation_writer.write(Deformation::SphereCarve {
        center: INITIAL_AREA_CENTER,
        radius: INITIAL_AREA_RADIUS,
    });
    *dispatched = true;
}
