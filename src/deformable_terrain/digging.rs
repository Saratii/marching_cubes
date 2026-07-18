use std::sync::Arc;

use bevy::{camera::primitives::MeshAabb, ecs::system::SystemParam, prelude::*};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use rustc_hash::FxHashMap;

use crate::{
    constants::{
        CHUNK_WORLD_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM,
        SAMPLES_PER_CHUNK_DIM_PADDED, SAMPLES_PER_CHUNK_PADDED, VOXEL_WORLD_SIZE,
    },
    conversions::{
        chunk_coord_to_world_pos, flatten_index, world_pos_to_chunk_coord, world_pos_to_voxel_index,
    },
    deformable_terrain::{
        chunk_entity_map::ChunkEntityMap,
        chunk_generator::{MaterialCode, dequantize_i16_to_f32, quantize_f32_to_i16},
        driver::{TerrainChunkMap, WriteCmd, WriteCmdSender},
        marching_cubes::mc::mc_mesh_generation,
        plugin::{ChunkTag, Deformation, Uniformity},
        sparse_voxel_octree::sphere_intersects_aabb,
        terrain::{
            NonUniformTerrainChunk, TerrainChunk, TerrainMaterialHandle, generate_bevy_mesh,
        },
    },
    player::player::MainCameraTag,
    ui::{configurable_settings::ConfigurableSettings, menu::MenuRoot},
};

/// Dig strength (configurable in the menu) is the world units the dug surface
/// advances per brush application at the dig center; the advance tapers
/// quadratically to zero at the sphere edge, which is what shapes a partial
/// dig into a smooth crater instead of a flat-floored cylinder. Effective
/// center dig speed is dig_strength / DIG_TIMER world units per second while
/// held.
const DIG_TIMER: f32 = 0.004; // seconds

/// How far past a brush's nominal radius its edits may reach, in world units.
/// Samples farther than `radius + BRUSH_INFLUENCE_MARGIN` from the dig center
/// are never written. Without this cap, the max()-style SDF update legitimately
/// changes values up to `radius + 10` away (the storage clamp), far outside the
/// gathered chunk set, leaving stale values in ungathered neighbors that then
/// disagree with their dug counterparts at chunk borders.
/// `chunk_coords_in_sphere` widens its gather bound by the same constant, so
/// every sample a dig can write lives in a gathered chunk by construction.
const BRUSH_INFLUENCE_MARGIN: f32 = 10.0 * VOXEL_WORLD_SIZE;

/// Depth below the brush surface (in world units) where the brush stops being
/// an exact sphere SDF and steepens into the skirt. Must exceed the mesher's
/// normal-gradient sampling reach (~2 voxels around a surface vertex) so the
/// slope kink is never visible on a freshly dug surface.
const BRUSH_SKIRT_START: f32 = 3.0 * VOXEL_WORLD_SIZE;

/// Slope of the skirt ramp, chosen so the brush reaches the bottom of the ±10
/// storage band exactly at BRUSH_INFLUENCE_MARGIN, where the update becomes an
/// exact no-op.
const BRUSH_SKIRT_SLOPE: f32 =
    (10.0 + VOXEL_WORLD_SIZE - BRUSH_SKIRT_START) / (BRUSH_INFLUENCE_MARGIN - BRUSH_SKIRT_START);

/// Bounded brush field: an exact negated sphere SDF near the surface, then a
/// steep but CONTINUOUS ramp down to below the storage clamp at the influence
/// margin. Truncating the brush with a hard distance cutoff instead leaves a
/// value cliff buried below the dug surface (fine on its own — nothing reads
/// there), but the next overlapping dig folds that cliff into ITS surface
/// values and meshes a jagged staircase along the old boundary. Continuity of
/// the stored field under any dig sequence is the invariant that prevents it.
#[inline(always)]
fn brush_sdf(sphere_sdf: f32) -> f32 {
    if sphere_sdf <= BRUSH_SKIRT_START {
        -sphere_sdf
    } else {
        -BRUSH_SKIRT_START - (sphere_sdf - BRUSH_SKIRT_START) * BRUSH_SKIRT_SLOPE
    }
}

#[derive(SystemParam)]
pub struct TerrainIo<'w> {
    pub terrain_chunk_map: ResMut<'w, TerrainChunkMap>,
    pub chunk_entity_map: ResMut<'w, ChunkEntityMap>,
}

pub(crate) fn chunk_coords_in_sphere(
    center: Vec3,
    radius: f32,
) -> impl Iterator<Item = (i16, i16, i16)> {
    let effective_radius = radius + BRUSH_INFLUENCE_MARGIN + VOXEL_WORLD_SIZE;
    let radius_squared = effective_radius * effective_radius;
    let min_world = center - Vec3::splat(effective_radius);
    let max_world = center + Vec3::splat(effective_radius);
    let min_chunk = world_pos_to_chunk_coord(&min_world);
    let max_chunk = world_pos_to_chunk_coord(&max_world);
    (min_chunk.0..=max_chunk.0).flat_map(move |chunk_x| {
        (min_chunk.1..=max_chunk.1).flat_map(move |chunk_y| {
            (min_chunk.2..=max_chunk.2).filter_map(move |chunk_z| {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
                let node_min = Vec3::new(
                    chunk_center.x - HALF_CHUNK - VOXEL_WORLD_SIZE,
                    chunk_center.y - HALF_CHUNK - VOXEL_WORLD_SIZE,
                    chunk_center.z - HALF_CHUNK - VOXEL_WORLD_SIZE,
                );
                let node_max = node_min + Vec3::splat(CHUNK_WORLD_SIZE + 2.0 * VOXEL_WORLD_SIZE);
                sphere_intersects_aabb(&center, radius_squared, &node_min, &node_max)
                    .then_some(chunk_coord)
            })
        })
    })
}

fn read_chunk_for_deform(
    terrain_chunk: &TerrainChunk,
    chunk_coord: (i16, i16, i16),
    map: &FxHashMap<(i16, i16, i16), TerrainChunk>,
) -> (Arc<[i16]>, Arc<[MaterialCode]>, Uniformity) {
    // Uniform chunks materialize to exactly the value the generator stores
    // for clamped samples: quantize(±10.0), NOT i16::MIN/MAX. i16::MIN is
    // one quantum below quantize(-10.0), so using it makes every stored copy
    // of a sample shared with a generated neighbor disagree by one quantum.
    match terrain_chunk {
        TerrainChunk::UniformAir => (
            build_uniform_padded_with_real_borders(quantize_f32_to_i16(10.0), chunk_coord, map),
            Arc::new([MaterialCode::Air; SAMPLES_PER_CHUNK]),
            Uniformity::Air,
        ),
        TerrainChunk::UniformDirt => (
            build_uniform_padded_with_real_borders(quantize_f32_to_i16(-10.0), chunk_coord, map),
            Arc::new([MaterialCode::Dirt; SAMPLES_PER_CHUNK]),
            Uniformity::Dirt,
        ),
        TerrainChunk::NonUniformTerrainChunk(chunk) => (
            Arc::clone(&chunk.densities),
            Arc::clone(&chunk.materials),
            Uniformity::NonUniform,
        ),
    }
}

/// World coordinate of a padded sample along one axis, derived from the
/// sample's global half-voxel index (an exact integer). Every chunk that
/// shares this sample computes the same integer, so the resulting f32 is
/// bit-identical regardless of which chunk computes it.
#[inline(always)]
fn padded_axis_world_coord(chunk_coord: i16, padded_idx: usize) -> f32 {
    const HALF_VOXEL: f32 = VOXEL_WORLD_SIZE * 0.5;
    let half_steps = chunk_coord as i32 * 2 * (SAMPLES_PER_CHUNK_DIM as i32 - 1)
        + 2 * padded_idx as i32
        - (SAMPLES_PER_CHUNK_DIM as i32 - 1)
        - 2;
    half_steps as f32 * HALF_VOXEL
}

fn axis_pairs(offset: i16, dim: usize) -> Vec<(usize, usize)> {
    match offset {
        1 => vec![(dim - 1, 2), (dim - 2, 1)],
        -1 => vec![(0, dim - 3), (1, dim - 2)],
        _ => (0..dim).map(|i| (i, i)).collect(),
    }
}

pub fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    camera: Query<(&Camera, &GlobalTransform), With<MainCameraTag>>,
    window: Query<&Window>,
    mut dig_timer: Local<f32>,
    time: Res<Time>,
    terrain_chunk_map: Res<TerrainChunkMap>,
    menu_root_query: Query<&MenuRoot>,
    mut deformation_writer: MessageWriter<Deformation>,
    settings: Res<ConfigurableSettings>,
) {
    if !menu_root_query.is_empty() {
        return;
    }
    let should_dig = if mouse_input.pressed(MouseButton::Left) {
        *dig_timer += time.delta_secs();
        if *dig_timer >= DIG_TIMER {
            *dig_timer = 0.0;
            true
        } else {
            false
        }
    } else {
        *dig_timer = 0.0;
        false
    };
    if should_dig {
        if let Some(cursor_pos) = window.iter().next().unwrap().cursor_position() {
            let (camera, camera_transform) = camera.iter().next().unwrap();
            if let Some(world_pos) =
                screen_to_world_ray(cursor_pos, camera, camera_transform, &terrain_chunk_map)
            {
                deformation_writer.write(Deformation::Sphere {
                    center: world_pos,
                    radius: settings.dig_radius,
                    strength: settings.dig_strength,
                });
            }
        }
    }
}

pub(crate) fn deformation_message_reader(
    mut deformation_reader: MessageReader<Deformation>,
    mut commands: Commands,
    material_handle: Res<TerrainMaterialHandle>,
    mut solid_chunk_query: Query<(&mut Collider, &mut Mesh3d), With<ChunkTag>>,
    mut mesh_handles: ResMut<Assets<Mesh>>,
    mut terrain_io: TerrainIo,
    write_cmd_sender: Res<WriteCmdSender>,
) {
    for deformation in deformation_reader.read() {
        let modified_chunks = match *deformation {
            Deformation::Sphere {
                center,
                radius,
                strength,
            } => dig_sphere(center, radius, strength, &terrain_io.terrain_chunk_map),
            Deformation::SphereCarve { center, radius } => {
                dig_sphere(center, radius, f32::INFINITY, &terrain_io.terrain_chunk_map)
            }
        };
        apply_modified_chunks(
            modified_chunks,
            &mut commands,
            &material_handle,
            &mut solid_chunk_query,
            &mut mesh_handles,
            &mut terrain_io,
            &write_cmd_sender,
        );
    }
}

fn apply_modified_chunks(
    modified_chunks: Vec<((i16, i16, i16), Arc<[i16]>, Arc<[MaterialCode]>, Uniformity)>,
    commands: &mut Commands,
    material_handle: &TerrainMaterialHandle,
    solid_chunk_query: &mut Query<(&mut Collider, &mut Mesh3d), With<ChunkTag>>,
    mesh_handles: &mut Assets<Mesh>,
    terrain_io: &mut TerrainIo,
    write_cmd_sender: &WriteCmdSender,
) {
    for (chunk_coord, densities, materials, uniformity) in modified_chunks {
        let entity = terrain_io.chunk_entity_map.get_option(chunk_coord);
        let (vertices, normals, material_ids, indices) = mc_mesh_generation(
            &densities,
            &materials,
            SAMPLES_PER_CHUNK_DIM,
            true,
            &densities,
        );
        let _ = write_cmd_sender.0.send(WriteCmd::UpdateNonUniform {
            densities: Arc::clone(&densities),
            materials: Arc::clone(&materials),
            chunk_coord,
        });
        match uniformity {
            Uniformity::Air => {
                let _ = write_cmd_sender
                    .0
                    .send(WriteCmd::RemoveUniformAir { chunk_coord });
            }
            Uniformity::Dirt => {
                let _ = write_cmd_sender
                    .0
                    .send(WriteCmd::RemoveUniformDirt { chunk_coord });
            }
            Uniformity::NonUniform => {}
            Uniformity::Unknown => unreachable!(),
        }
        let new_mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
        if new_mesh.count_vertices() > 0 {
            let collider = Collider::from_bevy_mesh(
                &new_mesh,
                &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
            )
            .unwrap();
            match entity {
                Some((entity, mesh_handle)) => {
                    let (mut collider_component, mut mesh) =
                        solid_chunk_query.get_mut(*entity).unwrap();
                    *collider_component = collider;
                    mesh_handles.remove(mesh_handle);
                    if let Some(aabb) = new_mesh.compute_aabb() {
                        commands.entity(*entity).insert(aabb);
                    }
                    let new_mesh_handle = mesh_handles.add(new_mesh);
                    *mesh = Mesh3d(new_mesh_handle.clone());
                    terrain_io
                        .chunk_entity_map
                        .replace_mesh_handle(chunk_coord, new_mesh_handle);
                }
                None => {
                    let new_mesh_handle = mesh_handles.add(new_mesh);
                    let new_entity = commands
                        .spawn((
                            collider,
                            Mesh3d(new_mesh_handle.clone()),
                            MeshMaterial3d(material_handle.0.clone()),
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                        ))
                        .id();
                    terrain_io
                        .chunk_entity_map
                        .insert(chunk_coord, (new_entity, new_mesh_handle));
                }
            }
        } else {
            if let Some((entity, mesh_handle)) = entity {
                commands.entity(*entity).despawn();
                mesh_handles.remove(mesh_handle);
                terrain_io.chunk_entity_map.remove(chunk_coord);
            }
        }
        let mut terrain_chunk_map_lock = terrain_io.terrain_chunk_map.0.lock().unwrap();
        terrain_chunk_map_lock.insert(
            chunk_coord,
            TerrainChunk::NonUniformTerrainChunk(NonUniformTerrainChunk {
                densities,
                materials,
            }),
        );
    }
}

/// Gather every chunk the brush can touch, apply the brush to each, and return
/// the chunks whose densities actually changed. `max_step` bounds how far the
/// field may rise per application: the dig strength setting for gradual player
/// digging, f32::INFINITY for an exact one-shot carve.
///
/// If any chunk in range is not loaded, the whole deformation is skipped —
/// digging half a sphere would leave shared border samples disagreeing with
/// the neighbor generated later.
fn dig_sphere(
    center: Vec3,
    radius: f32,
    max_step: f32,
    terrain_chunk_map: &TerrainChunkMap,
) -> Vec<((i16, i16, i16), Arc<[i16]>, Arc<[MaterialCode]>, Uniformity)> {
    let map_lock = terrain_chunk_map.0.lock().unwrap();
    let mut chunks = Vec::new();
    for chunk_coord in chunk_coords_in_sphere(center, radius) {
        let Some(terrain_chunk) = map_lock.get(&chunk_coord) else {
            warn!("skipping dig at {center}: chunk {chunk_coord:?} is not loaded");
            return Vec::new();
        };
        let (densities, materials, uniformity) =
            read_chunk_for_deform(terrain_chunk, chunk_coord, &map_lock);
        chunks.push((chunk_coord, densities, materials, uniformity));
    }
    drop(map_lock);
    chunks.retain_mut(|(chunk_coord, densities, ..)| {
        apply_brush_to_chunk(
            Arc::make_mut(densities),
            chunk_coord,
            center,
            radius,
            max_step,
        )
    });
    chunks
}

/// Raise the stored field toward the brush, at most `step` world units per
/// application: new = max(old, min(brush, old + step)), where
/// step = max_step * (1 - d²/r²). The max() means digging only ever removes
/// material, the min() caps the advance at the sphere so repeated applications
/// converge to the exact carve (new = max(old, brush)), and the quadratic
/// falloff digs fastest at the center so a partial dig is a smooth crater —
/// a constant step would lower the whole disk uniformly and leave a
/// flat-floored cylinder. The falloff reaches zero continuously at the sphere
/// edge, so the stored field stays continuous after any number of
/// applications.
fn apply_brush_to_chunk(
    densities: &mut [i16],
    chunk_coord: &(i16, i16, i16),
    dig_center: Vec3,
    radius: f32,
    max_step: f32,
) -> bool {
    let influence_cutoff = radius + BRUSH_INFLUENCE_MARGIN;
    let influence_cutoff_squared = influence_cutoff * influence_cutoff;
    let radius_squared = radius * radius;
    let mut chunk_modified = false;
    for z in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
        let world_z = padded_axis_world_coord(chunk_coord.2, z);
        for y in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
            let world_y = padded_axis_world_coord(chunk_coord.1, y);
            for x in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
                let world_x = padded_axis_world_coord(chunk_coord.0, x);
                let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                let distance_squared = voxel_world_pos.distance_squared(dig_center);
                if distance_squared > influence_cutoff_squared {
                    continue;
                }
                let step = if max_step.is_finite() {
                    let falloff = (1.0 - distance_squared / radius_squared).max(0.0);
                    if falloff == 0.0 {
                        continue;
                    }
                    max_step * falloff
                } else {
                    // exact carve: writes the full brush everywhere, including
                    // the skirt band just outside the sphere
                    f32::INFINITY
                };
                let brush = brush_sdf(distance_squared.sqrt() - radius);
                let flat_index =
                    flatten_index(x as u32, y as u32, z as u32, SAMPLES_PER_CHUNK_DIM_PADDED);
                let current_density = &mut densities[flat_index as usize];
                let old = dequantize_i16_to_f32(*current_density);
                let new_sdf = old.max(brush.min(old + step)).clamp(-10.0, 10.0);
                let new_quantized = quantize_f32_to_i16(new_sdf);
                if new_quantized != *current_density {
                    *current_density = new_quantized;
                    chunk_modified = true;
                }
            }
        }
    }
    chunk_modified
}

fn screen_to_world_ray(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    terrain_chunk_map: &TerrainChunkMap,
) -> Option<Vec3> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let ray_origin = ray.origin;
    let max_distance = 8.0;
    let step_size = 0.05;
    let mut distance_traveled = 0.0;
    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray.direction * distance_traveled;
        let chunk_coord = world_pos_to_chunk_coord(&current_pos);
        if let Some(chunk_data) = terrain_chunk_map.0.lock().unwrap().get(&chunk_coord) {
            let voxel_idx = world_pos_to_voxel_index(&current_pos, &chunk_coord);
            if chunk_data.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
                return Some(current_pos);
            }
            distance_traveled += step_size;
        }
    }
    None
}

/// Padded density buffer for a uniform chunk about to be dug: flat everywhere
/// except the sample planes shared with already-non-uniform neighbors, which
/// are copied from the neighbor so the dug result agrees with them exactly.
fn build_uniform_padded_with_real_borders(
    flat_value: i16,
    chunk_coord: (i16, i16, i16),
    map: &FxHashMap<(i16, i16, i16), TerrainChunk>,
) -> Arc<[i16]> {
    let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
    let mut densities = vec![flat_value; SAMPLES_PER_CHUNK_PADDED];
    for dx in -1..=1i16 {
        for dy in -1..=1i16 {
            for dz in -1..=1i16 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let neighbor_coord = (chunk_coord.0 + dx, chunk_coord.1 + dy, chunk_coord.2 + dz);
                let Some(TerrainChunk::NonUniformTerrainChunk(neighbor)) = map.get(&neighbor_coord)
                else {
                    continue;
                };
                for &(ox, nx) in &axis_pairs(dx, dim) {
                    for &(oy, ny) in &axis_pairs(dy, dim) {
                        for &(oz, nz) in &axis_pairs(dz, dim) {
                            let our_idx = flatten_index(ox as u32, oy as u32, oz as u32, dim);
                            let their_idx = flatten_index(nx as u32, ny as u32, nz as u32, dim);
                            densities[our_idx as usize] = neighbor.densities[their_idx as usize];
                        }
                    }
                }
            }
        }
    }
    Arc::from(densities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn build_map(coords: impl IntoIterator<Item = (i16, i16, i16)>) -> TerrainChunkMap {
        TerrainChunkMap(Arc::new(Mutex::new(
            coords
                .into_iter()
                .map(|c| (c, TerrainChunk::UniformDirt))
                .collect(),
        )))
    }

    fn dirt_materials() -> Arc<[MaterialCode]> {
        Arc::new([MaterialCode::Dirt; SAMPLES_PER_CHUNK])
    }

    fn cube_range(radius: i16) -> impl Iterator<Item = (i16, i16, i16)> + Clone {
        (-radius..=radius).flat_map(move |x| {
            (-radius..=radius).flat_map(move |y| (-radius..=radius).map(move |z| (x, y, z)))
        })
    }

    fn commit_modified(
        modified: &[((i16, i16, i16), Arc<[i16]>, Arc<[MaterialCode]>, Uniformity)],
        map: &TerrainChunkMap,
    ) {
        let mut lock = map.0.lock().unwrap();
        for (coord, densities, materials, _) in modified {
            lock.insert(
                *coord,
                TerrainChunk::NonUniformTerrainChunk(NonUniformTerrainChunk {
                    densities: Arc::clone(densities),
                    materials: Arc::clone(materials),
                }),
            );
        }
    }

    /// All non-uniform chunks currently stored in the map, in the shape the
    /// consistency checker expects.
    fn stored_nonuniform_chunks(
        map: &TerrainChunkMap,
    ) -> Vec<((i16, i16, i16), Arc<[i16]>, Arc<[MaterialCode]>, Uniformity)> {
        map.0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(coord, chunk)| match chunk {
                TerrainChunk::NonUniformTerrainChunk(c) => Some((
                    *coord,
                    Arc::clone(&c.densities),
                    Arc::clone(&c.materials),
                    Uniformity::NonUniform,
                )),
                _ => None,
            })
            .collect()
    }

    /// Every sample plane shared by two chunks must hold identical values in
    /// both copies — the invariant that keeps meshes crack-free at borders.
    fn assert_padding_walls_consistent(
        chunks: &[((i16, i16, i16), Arc<[i16]>, Arc<[MaterialCode]>, Uniformity)],
    ) {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let by_coord: FxHashMap<(i16, i16, i16), &Arc<[i16]>> =
            chunks.iter().map(|(c, d, ..)| (*c, d)).collect();
        for (coord, densities, ..) in chunks {
            for dx in -1..=1i16 {
                for dy in -1..=1i16 {
                    for dz in -1..=1i16 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let neighbor_coord = (coord.0 + dx, coord.1 + dy, coord.2 + dz);
                        let Some(neighbor_densities) = by_coord.get(&neighbor_coord) else {
                            continue;
                        };
                        for &(our_x, their_x) in &axis_pairs(dx, dim) {
                            for &(our_y, their_y) in &axis_pairs(dy, dim) {
                                for &(our_z, their_z) in &axis_pairs(dz, dim) {
                                    let our_idx = flatten_index(
                                        our_x as u32,
                                        our_y as u32,
                                        our_z as u32,
                                        dim,
                                    );
                                    let their_idx = flatten_index(
                                        their_x as u32,
                                        their_y as u32,
                                        their_z as u32,
                                        dim,
                                    );
                                    assert_eq!(
                                        densities[our_idx as usize],
                                        neighbor_densities[their_idx as usize],
                                        "padding mismatch between {coord:?} and {neighbor_coord:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn assert_carve_surface_near_radius(
        chunk_coord: (i16, i16, i16),
        densities: &[i16],
        center: Vec3,
        radius: f32,
    ) {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let tolerance = 2.0 * VOXEL_WORLD_SIZE;
        let mut found_crossing = false;
        for x in 2..dim - 2 {
            let world_x = padded_axis_world_coord(chunk_coord.0, x);
            for y in 2..dim - 2 {
                let world_y = padded_axis_world_coord(chunk_coord.1, y);
                for z in 2..dim - 3 {
                    let idx_a = flatten_index(x as u32, y as u32, z as u32, dim) as usize;
                    let idx_b = flatten_index(x as u32, y as u32, (z + 1) as u32, dim) as usize;
                    let a = dequantize_i16_to_f32(densities[idx_a]);
                    let b = dequantize_i16_to_f32(densities[idx_b]);
                    if (a < 0.0) == (b < 0.0) {
                        continue;
                    }
                    found_crossing = true;
                    let t = a / (a - b);
                    let z_a = padded_axis_world_coord(chunk_coord.2, z);
                    let z_b = padded_axis_world_coord(chunk_coord.2, z + 1);
                    let crossing = Vec3::new(world_x, world_y, z_a + t * (z_b - z_a));
                    let distance = crossing.distance(center);
                    assert!(
                        (distance - radius).abs() <= tolerance,
                        "surface crossing at {crossing:?} was distance {distance} from center, expected close to radius {radius}"
                    );
                }
            }
        }
        assert!(
            found_crossing,
            "expected at least one sign change on the carved surface for {chunk_coord:?}"
        );
    }

    fn assert_normals_point_toward_center(
        chunk_coord: (i16, i16, i16),
        densities: &Arc<[i16]>,
        materials: &Arc<[MaterialCode]>,
        center: Vec3,
    ) {
        let (vertices, normals, _material_ids, _indices) =
            mc_mesh_generation(densities, materials, SAMPLES_PER_CHUNK_DIM, true, densities);
        assert!(
            !vertices.is_empty(),
            "expected a carved surface to be meshed"
        );
        let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
        for (vertex, normal) in vertices.iter().zip(normals.iter()) {
            let world_vertex = chunk_center + *vertex;
            let to_center = (center - world_vertex).normalize_or_zero();
            if to_center == Vec3::ZERO {
                continue;
            }
            let alignment = normal.normalize_or_zero().dot(to_center);
            assert!(
                alignment > 0.5,
                "expected normal {normal:?} at {world_vertex:?} to point toward the dig center, alignment was {alignment}"
            );
        }
    }

    #[test]
    fn carve_meshes_a_sphere_surface_at_the_dig_radius() {
        let chunk_coord = (0, 0, 0);
        let center = chunk_coord_to_world_pos(&chunk_coord);
        let radius = 0.25 * CHUNK_WORLD_SIZE;
        let map = build_map([chunk_coord]);

        let modified = dig_sphere(center, radius, f32::INFINITY, &map);

        assert_eq!(modified.len(), 1);
        let (coord, densities, materials, uniformity) = &modified[0];
        assert_eq!(*coord, chunk_coord);
        assert!(matches!(uniformity, Uniformity::Dirt));
        assert_carve_surface_near_radius(chunk_coord, densities, center, radius);
        assert_normals_point_toward_center(chunk_coord, densities, materials, center);
    }

    #[test]
    fn repeated_digs_converge_to_the_exact_carve_inside_the_sphere() {
        let chunk_coord = (0, 0, 0);
        let center = chunk_coord_to_world_pos(&chunk_coord);
        let radius = 0.25 * CHUNK_WORLD_SIZE;
        let strength = 0.4;
        // Samples with falloff >= 0.5 (d^2 <= r^2 / 2) advance at least
        // strength / 2 per application until they hit the brush cap; the
        // deepest deficit is radius + 10 (storage clamp) world units.
        let applications = ((radius + 10.0) / (strength * 0.5)).ceil() as usize + 2;

        let dug_map = build_map([chunk_coord]);
        for _ in 0..applications {
            let modified = dig_sphere(center, radius, strength, &dug_map);
            commit_modified(&modified, &dug_map);
        }

        let carved_map = build_map([chunk_coord]);
        let carved = dig_sphere(center, radius, f32::INFINITY, &carved_map);
        assert_eq!(carved.len(), 1);

        let dug = stored_nonuniform_chunks(&dug_map);
        assert_eq!(dug.len(), 1);
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let mut compared = 0;
        for x in 0..dim {
            for y in 0..dim {
                for z in 0..dim {
                    let world = Vec3::new(
                        padded_axis_world_coord(chunk_coord.0, x),
                        padded_axis_world_coord(chunk_coord.1, y),
                        padded_axis_world_coord(chunk_coord.2, z),
                    );
                    if world.distance_squared(center) > 0.5 * radius * radius {
                        continue;
                    }
                    let idx = flatten_index(x as u32, y as u32, z as u32, dim) as usize;
                    assert_eq!(
                        dug[0].1[idx], carved[0].1[idx],
                        "converged dig should equal the exact carve at ({x}, {y}, {z})"
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 0, "no samples inside the sphere were compared");
    }

    /// Flat ground at world y = 0: density = clamp(world_y, ±10).
    fn flat_ground_chunk(coord: (i16, i16, i16)) -> TerrainChunk {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let mut densities = vec![0i16; SAMPLES_PER_CHUNK_PADDED];
        for y in 0..dim {
            let q = quantize_f32_to_i16(padded_axis_world_coord(coord.1, y).clamp(-10.0, 10.0));
            for z in 0..dim {
                for x in 0..dim {
                    densities[flatten_index(x as u32, y as u32, z as u32, dim) as usize] = q;
                }
            }
        }
        TerrainChunk::NonUniformTerrainChunk(NonUniformTerrainChunk {
            densities: Arc::from(densities),
            materials: Arc::new([MaterialCode::Dirt; SAMPLES_PER_CHUNK]),
        })
    }

    /// Interpolated air->solid crossing height in one sample column of a
    /// chunk at cy = 0, the same way marching cubes places vertices.
    fn surface_height(densities: &[i16], x: usize, z: usize) -> f32 {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        for y in (0..dim - 1).rev() {
            let upper = dequantize_i16_to_f32(
                densities[flatten_index(x as u32, (y + 1) as u32, z as u32, dim) as usize],
            );
            let lower = dequantize_i16_to_f32(
                densities[flatten_index(x as u32, y as u32, z as u32, dim) as usize],
            );
            if upper >= 0.0 && lower < 0.0 {
                let t = upper / (upper - lower);
                let y_up = padded_axis_world_coord(0, y + 1);
                let y_low = padded_axis_world_coord(0, y);
                return y_up + t * (y_low - y_up);
            }
        }
        panic!("no surface crossing in column ({x}, {z})");
    }

    /// Regression test for the "shallow wide cylinder" bug: a partial dig
    /// (released before converging) must taper smoothly from deepest at the
    /// center to untouched at the sphere edge. A constant per-application
    /// step instead lowers the whole disk uniformly, leaving a flat floor
    /// with a sheer circular wall.
    #[test]
    fn partial_dig_leaves_a_smooth_crater_not_a_cylinder() {
        let radius = 4.0;
        let strength = 0.4;
        let applications = 5;
        let map = TerrainChunkMap(Arc::new(Mutex::new(
            cube_range(1).map(|c| (c, flat_ground_chunk(c))).collect(),
        )));
        let chunk_center = chunk_coord_to_world_pos(&(0, 0, 0));
        let center = Vec3::new(chunk_center.x, 0.0, chunk_center.z);

        for _ in 0..applications {
            let modified = dig_sphere(center, radius, strength, &map);
            commit_modified(&modified, &map);
        }

        let lock = map.0.lock().unwrap();
        let Some(TerrainChunk::NonUniformTerrainChunk(chunk)) = lock.get(&(0, 0, 0)) else {
            panic!("dug chunk should be non-uniform");
        };
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let closest_idx = |target: f32, chunk_coord: i16| {
            (0..dim)
                .min_by(|&a, &b| {
                    let da = (padded_axis_world_coord(chunk_coord, a) - target).abs();
                    let db = (padded_axis_world_coord(chunk_coord, b) - target).abs();
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
        };
        let center_x = closest_idx(center.x, 0);
        let center_z = closest_idx(center.z, 0);

        let center_depth = -surface_height(&chunk.densities, center_x, center_z);
        assert!(
            center_depth > 1.0,
            "expected a meaningful crater at the center, depth was {center_depth}"
        );

        // walk outward along +x: depth must taper monotonically to zero with
        // no wall (no big drop between adjacent columns)
        let mut previous_depth = center_depth;
        for x in center_x..dim {
            let depth = -surface_height(&chunk.densities, x, center_z);
            assert!(
                depth <= previous_depth + 0.05,
                "crater depth increased moving outward at column {x}: {previous_depth} -> {depth}"
            );
            assert!(
                previous_depth - depth <= 0.5,
                "sheer wall between adjacent columns at {x}: {previous_depth} -> {depth}"
            );
            let outward = padded_axis_world_coord(0, x) - center.x;
            if outward > radius {
                assert!(
                    depth.abs() <= 0.1,
                    "ground beyond the dig radius should be untouched, depth at distance {outward} was {depth}"
                );
            }
            previous_depth = depth;
        }
    }

    #[test]
    fn multi_chunk_dig_keeps_shared_border_samples_consistent() {
        // dig centered on the corner where 8 chunks meet
        let center = chunk_coord_to_world_pos(&(0, 0, 0)) + Vec3::splat(HALF_CHUNK);
        let radius = 0.25 * CHUNK_WORLD_SIZE;
        let map = build_map(cube_range(1));

        for _ in 0..3 {
            let modified = dig_sphere(center, radius, 0.4, &map);
            assert!(
                modified.len() >= 8,
                "a corner dig should modify all 8 chunks sharing the corner, got {}",
                modified.len()
            );
            commit_modified(&modified, &map);
            assert_padding_walls_consistent(&stored_nonuniform_chunks(&map));
        }
    }

    #[test]
    fn digging_a_uniform_chunk_agrees_with_its_nonuniform_neighbor() {
        // (1, 0, 0) already dug once (non-uniform), (0, 0, 0) still uniform;
        // a dig straddling their border must leave every shared sample equal,
        // which requires the uniform chunk to pick up the neighbor's real
        // border values before the brush is applied.
        let map = build_map(cube_range(1));
        let border = chunk_coord_to_world_pos(&(0, 0, 0)) + Vec3::new(HALF_CHUNK, 0.0, 0.0);
        let radius = 2.0;

        let first = dig_sphere(border + Vec3::X * radius, radius, 0.4, &map);
        assert!(first.iter().any(|(c, ..)| *c == (1, 0, 0)));
        commit_modified(&first, &map);

        let second = dig_sphere(border, radius, 0.4, &map);
        assert!(second.iter().any(|(c, ..)| *c == (0, 0, 0)));
        commit_modified(&second, &map);

        assert_padding_walls_consistent(&stored_nonuniform_chunks(&map));
    }

    #[test]
    fn dig_is_skipped_when_a_chunk_in_range_is_not_loaded() {
        let map = build_map([(0, 0, 0)]);
        let near_border = chunk_coord_to_world_pos(&(0, 0, 0)) + Vec3::new(HALF_CHUNK, 0.0, 0.0);

        let modified = dig_sphere(near_border, 2.0, 0.4, &map);

        assert!(
            modified.is_empty(),
            "a dig reaching an unloaded chunk must be skipped entirely"
        );
    }

    /// Ground at world y = GROUND_HEIGHT stored the way the driver stores it:
    /// chunks entirely >= 10 below the surface as TerrainChunk::UniformDirt,
    /// entirely >= 10 above as UniformAir, and buffers only near the surface.
    const GROUND_HEIGHT: f32 = 5.0;

    fn ground_stack_chunk(coord: (i16, i16, i16)) -> TerrainChunk {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let lowest = padded_axis_world_coord(coord.1, 0) - GROUND_HEIGHT;
        let highest = padded_axis_world_coord(coord.1, dim - 1) - GROUND_HEIGHT;
        if highest <= -10.0 {
            return TerrainChunk::UniformDirt;
        }
        if lowest >= 10.0 {
            return TerrainChunk::UniformAir;
        }
        let mut densities = vec![0i16; SAMPLES_PER_CHUNK_PADDED];
        for y in 0..dim {
            let world_y = padded_axis_world_coord(coord.1, y);
            let q = quantize_f32_to_i16((world_y - GROUND_HEIGHT).clamp(-10.0, 10.0));
            for z in 0..dim {
                for x in 0..dim {
                    densities[flatten_index(x as u32, y as u32, z as u32, dim) as usize] = q;
                }
            }
        }
        TerrainChunk::NonUniformTerrainChunk(NonUniformTerrainChunk {
            densities: Arc::from(densities),
            materials: dirt_materials(),
        })
    }

    /// Straight-down replica of screen_to_world_ray's marching (same step and
    /// is_solid lookup), the way the player aims a dig at the receding floor.
    fn raycast_down(map: &TerrainChunkMap, x: f32, z: f32) -> Option<Vec3> {
        let lock = map.0.lock().unwrap();
        let mut y = 15.0;
        while y > -15.0 {
            let pos = Vec3::new(x, y, z);
            let chunk_coord = world_pos_to_chunk_coord(&pos);
            if let Some(chunk) = lock.get(&chunk_coord) {
                let voxel_idx = world_pos_to_voxel_index(&pos, &chunk_coord);
                if chunk.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
                    return Some(pos);
                }
            }
            y -= 0.05;
        }
        None
    }

    /// Topmost air->solid crossing per sample column across the whole stored
    /// stack, interpolated the same way marching cubes places vertices.
    /// Keyed by global sample column index.
    fn surface_heights(map: &TerrainChunkMap) -> FxHashMap<(i32, i32), f32> {
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        let lock = map.0.lock().unwrap();
        let mut heights = FxHashMap::default();
        for cx in -1i16..=2 {
            for cz in -1i16..=1 {
                for lx in 1..dim - 1 {
                    for lz in 1..dim - 1 {
                        let gx = cx as i32 * 63 + lx as i32 - 1;
                        let gz = cz as i32 * 63 + lz as i32 - 1;
                        if heights.contains_key(&(gx, gz)) {
                            continue;
                        }
                        'rows: for cy in [2i16, 1, 0, -1] {
                            let Some(TerrainChunk::NonUniformTerrainChunk(chunk)) =
                                lock.get(&(cx, cy, cz))
                            else {
                                continue;
                            };
                            for ly in (1..dim - 2).rev() {
                                let upper = dequantize_i16_to_f32(
                                    chunk.densities[flatten_index(
                                        lx as u32,
                                        (ly + 1) as u32,
                                        lz as u32,
                                        dim,
                                    ) as usize],
                                );
                                let lower = dequantize_i16_to_f32(
                                    chunk.densities[flatten_index(
                                        lx as u32, ly as u32, lz as u32, dim,
                                    ) as usize],
                                );
                                if upper >= 0.0 && lower < 0.0 {
                                    let t = upper / (upper - lower);
                                    let y_up = padded_axis_world_coord(cy, ly + 1);
                                    let y_low = padded_axis_world_coord(cy, ly);
                                    heights.insert((gx, gz), y_up + t * (y_low - y_up));
                                    break 'rows;
                                }
                            }
                        }
                    }
                }
            }
        }
        heights
    }

    /// The gameplay scenario that broke in-game: big overlapping digs aimed by
    /// raycast at the receding surface, punching through a mixed stack of
    /// UniformAir / surface / UniformDirt chunks. After every application all
    /// stored copies of shared border samples must agree; at the end the
    /// surface must be cliff-free (no chunk lagging its neighbor by world
    /// units) and the meshes of two adjacent dug chunks must weld exactly at
    /// their border.
    #[test]
    fn overlapping_big_digs_through_mixed_uniform_stack_stay_seamless() {
        let radius = 13.0;
        let strength = 0.4;
        let map = TerrainChunkMap(Arc::new(Mutex::new(
            (-1i16..=2)
                .flat_map(|x| {
                    (-1i16..=2).flat_map(move |y| {
                        (-1i16..=1).map(move |z| ((x, y, z), ground_stack_chunk((x, y, z))))
                    })
                })
                .collect(),
        )));
        {
            let lock = map.0.lock().unwrap();
            assert!(matches!(lock[&(0, -1, 0)], TerrainChunk::UniformDirt));
            assert!(matches!(lock[&(0, 2, 0)], TerrainChunk::UniformAir));
            assert!(matches!(
                lock[&(0, 0, 0)],
                TerrainChunk::NonUniformTerrainChunk(_)
            ));
        }

        for hole_x in [0.0f32, 8.0] {
            for application in 0..4 {
                let hit = raycast_down(&map, hole_x, 0.0)
                    .expect("the aiming ray should always find the surface");
                let modified = dig_sphere(hit, radius, strength, &map);
                assert!(
                    !modified.is_empty(),
                    "dig at x={hole_x} application {application} should modify chunks"
                );
                commit_modified(&modified, &map);
                assert_padding_walls_consistent(&stored_nonuniform_chunks(&map));
            }
        }

        {
            let lock = map.0.lock().unwrap();
            assert!(
                matches!(lock[&(0, -1, 0)], TerrainChunk::NonUniformTerrainChunk(_)),
                "the dig should have reached into the uniform-dirt row"
            );
        }

        // no cliff anywhere on the surface: a chunk that failed to dig or
        // reverted would lag its dug neighbor by world units at the border
        let heights = surface_heights(&map);
        let deepest = heights.values().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            deepest < GROUND_HEIGHT - 1.0,
            "expected a real crater, deepest surface was {deepest}"
        );
        for (&(gx, gz), &height) in &heights {
            for neighbor in [(gx + 1, gz), (gx, gz + 1)] {
                if let Some(&neighbor_height) = heights.get(&neighbor) {
                    assert!(
                        (height - neighbor_height).abs() <= 0.3,
                        "surface cliff between columns {:?} at {height} and {neighbor:?} at {neighbor_height}",
                        (gx, gz),
                    );
                }
            }
        }

        // mesh-level: vertices on the shared border plane of two dug chunks
        // must coincide, or the rendered surfaces visibly split at the border
        let (a_coord, b_coord) = ((0, 0, 0), (1, 0, 0));
        let world_vertices = |coord: (i16, i16, i16)| -> Vec<Vec3> {
            let lock = map.0.lock().unwrap();
            let TerrainChunk::NonUniformTerrainChunk(chunk) = &lock[&coord] else {
                panic!("expected dug chunk at {coord:?}");
            };
            let (vertices, ..) = mc_mesh_generation(
                &chunk.densities,
                &chunk.materials,
                SAMPLES_PER_CHUNK_DIM,
                true,
                &chunk.densities,
            );
            let center = chunk_coord_to_world_pos(&coord);
            vertices.into_iter().map(|v| center + v).collect()
        };
        let border_x = padded_axis_world_coord(a_coord.0, SAMPLES_PER_CHUNK_DIM_PADDED - 2);
        let on_plane = |vertices: &[Vec3]| -> Vec<Vec3> {
            vertices
                .iter()
                .copied()
                .filter(|v| (v.x - border_x).abs() < 1e-3)
                .collect()
        };
        let a_plane = on_plane(&world_vertices(a_coord));
        let b_plane = on_plane(&world_vertices(b_coord));
        assert!(
            !a_plane.is_empty() && !b_plane.is_empty(),
            "the crater wall should cross the {a_coord:?}|{b_coord:?} border"
        );
        for (own, other) in [(&a_plane, &b_plane), (&b_plane, &a_plane)] {
            for vertex in own.iter() {
                let closest = other
                    .iter()
                    .map(|v| v.distance(*vertex))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    closest <= 0.02,
                    "border vertex {vertex:?} has no counterpart in the neighbor mesh (closest {closest})"
                );
            }
        }
    }

    /// Overlapping digs must never leave a value cliff in the stored field:
    /// axis-adjacent samples staying within a small step is what keeps a later
    /// overlapping dig from meshing a jagged staircase along an old brush
    /// boundary. A true SDF steps one voxel (~0.19); the brush skirt may
    /// legitimately step a few times that, but nothing near the 10-unit band.
    #[test]
    fn overlapping_digs_leave_a_continuous_density_field() {
        let radius = 2.0;
        let first_center = chunk_coord_to_world_pos(&(0, 0, 0));
        let second_center = first_center + Vec3::new(radius * 1.25, 0.0, 0.0);
        let map = build_map(cube_range(1));

        for center in [first_center, second_center] {
            for _ in 0..3 {
                let modified = dig_sphere(center, radius, 0.4, &map);
                commit_modified(&modified, &map);
            }
        }

        let max_step = 2.0;
        let dim = SAMPLES_PER_CHUNK_DIM_PADDED;
        for (coord, densities, ..) in stored_nonuniform_chunks(&map) {
            for z in 0..dim {
                for y in 0..dim {
                    for x in 0..dim {
                        let here = dequantize_i16_to_f32(
                            densities[flatten_index(x as u32, y as u32, z as u32, dim) as usize],
                        );
                        for (nx, ny, nz) in [(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)] {
                            if nx >= dim || ny >= dim || nz >= dim {
                                continue;
                            }
                            let there = dequantize_i16_to_f32(
                                densities
                                    [flatten_index(nx as u32, ny as u32, nz as u32, dim) as usize],
                            );
                            assert!(
                                (here - there).abs() <= max_step,
                                "value cliff in chunk {coord:?} between ({x}, {y}, {z}) = {here} and ({nx}, {ny}, {nz}) = {there}"
                            );
                        }
                    }
                }
            }
        }
    }
}
