use std::sync::Arc;

use bevy::{camera::primitives::MeshAabb, ecs::system::SystemParam, prelude::*};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};

use crate::{
    conversions::{
        chunk_coord_to_world_pos, flatten_index, world_pos_to_chunk_coord, world_pos_to_voxel_index,
    },
    data_loader::{
        driver::{TerrainChunkMap, WriteCmd, WriteCmdSender},
        file_loader::ChunkEntityMap,
    },
    marching_cubes::mc::mc_mesh_generation,
    player::player::MainCameraTag,
    sparse_voxel_octree::sphere_intersects_aabb,
    terrain::{
        chunk_generator::{dequantize_i16_to_f32, quantize_f32_to_i16},
        terrain::{
            CHUNK_SIZE, ChunkTag, HALF_CHUNK, NonUniformTerrainChunk, SAMPLES_PER_CHUNK,
            SAMPLES_PER_CHUNK_DIM, TerrainChunk, TerrainMaterialHandle, Uniformity, VOXEL_SIZE,
            generate_bevy_mesh,
        },
    },
};

const DIG_STRENGTH: f32 = 0.5;
const DIG_TIMER: f32 = 0.004; // seconds
const DIG_RADIUS: f32 = 2.0; // world space
const DIG_RADIUS_SQUARED: f32 = DIG_RADIUS * DIG_RADIUS;

#[derive(SystemParam)]
pub struct TerrainIo<'w> {
    pub terrain_chunk_map: ResMut<'w, TerrainChunkMap>,
    pub chunk_entity_map: ResMut<'w, ChunkEntityMap>,
}
pub fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCameraTag>>,
    window: Single<&Window>,
    mut commands: Commands,
    mut dig_timer: Local<f32>,
    time: Res<Time>,
    material_handle: Res<TerrainMaterialHandle>,
    mut solid_chunk_query: Query<(&mut Collider, &mut Mesh3d, Entity), With<ChunkTag>>,
    mut mesh_handles: ResMut<Assets<Mesh>>,
    mut terrain_io: TerrainIo,
    write_cmd_sender: Res<WriteCmdSender>,
) {
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
        if let Some(cursor_pos) = window.cursor_position() {
            if let Some((world_pos, _, _)) = screen_to_world_ray(
                cursor_pos,
                camera.0,
                camera.1,
                &terrain_io.terrain_chunk_map,
            ) {
                let modified_chunks = dig_sphere(
                    world_pos,
                    DIG_RADIUS,
                    DIG_RADIUS_SQUARED,
                    DIG_STRENGTH,
                    &mut terrain_io.terrain_chunk_map,
                );
                for (chunk_coord, densities, materials, uniformity) in modified_chunks {
                    let entity = terrain_io.chunk_entity_map.0.get(&chunk_coord);
                    let (vertices, normals, material_ids, indices) = mc_mesh_generation(
                        &densities,
                        &materials,
                        SAMPLES_PER_CHUNK_DIM,
                        HALF_CHUNK,
                    );
                    match uniformity {
                        Uniformity::Air | Uniformity::Dirt => {
                            write_cmd_sender
                                .0
                                .send(WriteCmd::UpdateNonUniform {
                                    densities: Arc::clone(&densities),
                                    materials: Arc::clone(&materials),
                                    coord: chunk_coord,
                                })
                                .unwrap();
                            if uniformity == Uniformity::Air {
                                write_cmd_sender
                                    .0
                                    .send(WriteCmd::RemoveUniformAir { coord: chunk_coord })
                                    .unwrap();
                            } else {
                                write_cmd_sender
                                    .0
                                    .send(WriteCmd::RemoveUniformDirt { coord: chunk_coord })
                                    .unwrap();
                            }
                        }
                        Uniformity::NonUniform => {
                            write_cmd_sender
                                .0
                                .send(WriteCmd::UpdateNonUniform {
                                    densities: Arc::clone(&densities),
                                    materials: Arc::clone(&materials),
                                    coord: chunk_coord,
                                })
                                .unwrap();
                        }
                    }
                    let new_mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
                    if new_mesh.count_vertices() > 0 {
                        let collider = Collider::from_bevy_mesh(
                            &new_mesh,
                            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                        )
                        .unwrap();
                        match entity {
                            //entity already existed, update it
                            Some(entity) => {
                                let (mut collider_component, mut mesh_handle, _) =
                                    solid_chunk_query.get_mut(*entity).unwrap();
                                *collider_component = collider;
                                mesh_handles.remove(&mesh_handle.0);
                                if let Some(aabb) = new_mesh.compute_aabb() {
                                    commands.entity(*entity).insert(aabb);
                                }
                                *mesh_handle = Mesh3d(mesh_handles.add(new_mesh));
                            }
                            //entity did not already exist
                            None => {
                                let new_entity = commands
                                    .spawn((
                                        collider,
                                        Mesh3d(mesh_handles.add(new_mesh)),
                                        MeshMaterial3d(material_handle.0.clone()),
                                        ChunkTag,
                                        Transform::from_translation(chunk_coord_to_world_pos(
                                            &chunk_coord,
                                        )),
                                    ))
                                    .id();
                                terrain_io
                                    .chunk_entity_map
                                    .0
                                    .insert(chunk_coord, new_entity);
                            }
                        }
                    } else {
                        //no geometry, remove existing entity if it exists
                        if let Some(entity) = entity {
                            commands.entity(*entity).despawn();
                        }
                        terrain_io.chunk_entity_map.0.remove(&chunk_coord);
                    }
                    //replace chunks in chunk map
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
        }
    }
}

fn dig_sphere(
    center: Vec3,
    radius: f32,
    radius_squared: f32,
    strength: f32,
    terrain_chunk_map: &mut TerrainChunkMap,
) -> Vec<((i16, i16, i16), Arc<[i16]>, Arc<[u8]>, Uniformity)> {
    let mut modified_chunks = Vec::new();
    let min_world = center - Vec3::splat(radius);
    let max_world = center + Vec3::splat(radius);
    let min_chunk = world_pos_to_chunk_coord(&min_world);
    let max_chunk = world_pos_to_chunk_coord(&max_world);
    let inv_radius_sq = 1.0 / radius_squared;
    //collect copies of all modified chunks
    let mut terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_y in min_chunk.1..=max_chunk.1 {
            for chunk_z in min_chunk.2..=max_chunk.2 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
                let node_min = Vec3::new(
                    chunk_center.x - HALF_CHUNK,
                    chunk_center.y - HALF_CHUNK,
                    chunk_center.z - HALF_CHUNK,
                );
                let node_max = node_min + Vec3::splat(CHUNK_SIZE);
                if sphere_intersects_aabb(&center, radius_squared, &node_min, &node_max) {
                    let terrain_chunk = terrain_chunk_map_lock.get_mut(&chunk_coord).expect(
                        "During dig, tried to modify a chunk that does not exist in the terrain chunk map!",
                    );
                    let (densities, materials, uniformity): (Arc<[i16]>, Arc<[u8]>, Uniformity) =
                        match terrain_chunk {
                            TerrainChunk::UniformAir => (
                                Arc::new([i16::MAX; SAMPLES_PER_CHUNK]),
                                Arc::new([0u8; SAMPLES_PER_CHUNK]),
                                Uniformity::Air,
                            ),
                            TerrainChunk::UniformDirt => (
                                Arc::new([i16::MIN; SAMPLES_PER_CHUNK]),
                                Arc::new([1u8; SAMPLES_PER_CHUNK]),
                                Uniformity::Dirt,
                            ),
                            TerrainChunk::NonUniformTerrainChunk(chunk) => (
                                Arc::clone(&chunk.densities),
                                Arc::clone(&chunk.materials),
                                Uniformity::NonUniform,
                            ),
                        };
                    modified_chunks.push((chunk_coord, densities, materials, uniformity));
                }
            }
        }
    }
    drop(terrain_chunk_map_lock);
    modified_chunks.retain_mut(|(chunk_coord, densities, _, _)| {
        let dens_mut: &mut [i16] = Arc::make_mut(densities);
        modify_chunk_voxels(
            dens_mut,
            chunk_coord,
            center,
            radius_squared,
            strength,
            inv_radius_sq,
        )
    });
    modified_chunks
}

fn modify_chunk_voxels(
    densities: &mut [i16],
    chunk_coord: &(i16, i16, i16),
    dig_center: Vec3,
    radius_squared: f32,
    strength: f32,
    inv_radius_sq: f32,
) -> bool {
    let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
    let node_min = Vec3::new(
        chunk_center.x - HALF_CHUNK,
        chunk_center.y - HALF_CHUNK,
        chunk_center.z - HALF_CHUNK,
    );
    let mut chunk_modified = false;
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let world_z = node_min.z + z as f32 * VOXEL_SIZE;
        for y in 0..SAMPLES_PER_CHUNK_DIM {
            let world_y = node_min.y + y as f32 * VOXEL_SIZE;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let world_x = node_min.x + x as f32 * VOXEL_SIZE;
                let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                let distance_squared = voxel_world_pos.distance_squared(dig_center);
                if distance_squared <= radius_squared {
                    let falloff = 1.0 - distance_squared * inv_radius_sq;
                    let dig_amount = strength * falloff;
                    let flat_index =
                        flatten_index(x as u32, y as u32, z as u32, SAMPLES_PER_CHUNK_DIM);
                    let current_density = &mut densities[flat_index as usize];
                    if *current_density < 0 {
                        let sdf_f32 = dequantize_i16_to_f32(*current_density);
                        let new_sdf = (sdf_f32 + dig_amount).clamp(-10.0, 10.0);
                        *current_density = quantize_f32_to_i16(new_sdf);
                        chunk_modified = true;
                    }
                }
            }
        }
    }
    return chunk_modified;
}

fn screen_to_world_ray(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    terrain_chunk_map: &TerrainChunkMap,
) -> Option<(Vec3, Vec3, (i16, i16, i16))> {
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
            let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
            if chunk_data.is_solid(voxel_idx.0, voxel_idx.1, voxel_idx.2) {
                return Some((current_pos, chunk_world_pos, chunk_coord));
            }
            distance_traveled += step_size;
        } else {
            break;
        }
    }
    None
}
