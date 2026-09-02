use bevy::image::{ImageLoaderSettings, ImageSampler};
use bevy::{
    mesh::VertexAttributeValues,
    prelude::*,
    render::render_resource::{AddressMode, SamplerDescriptor},
};
use bevy_rapier3d::prelude::{Collider, ColliderMassProperties, Damping, RigidBody, Velocity};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::fs::{create_dir_all, read_to_string, write};

use crate::{
    constants::{SAMPLES_PER_CHUNK_DIM_PADDED, VOXEL_WORLD_SIZE},
    conversions::{world_pos_to_chunk_coord, world_pos_to_voxel_index},
    deformable_terrain::{
        chunk_entity_map::ChunkEntityMap,
        driver::TerrainChunkMap,
        file_loader::get_project_root,
        plugin::ChunkTag,
        terrain::TerrainChunk,
    },
};

/// Ore removed by digging is banked until it is worth a rock, so a held dig
/// drops a few chunky rocks instead of a stream of specks. A dig that stops
/// short of this leaves its ore banked for the next one.
const MIN_DEBRIS_VOLUME: f32 = 0.3;

/// Distinct rock shapes, picked round-robin so every rock is not the same lump.
const DEBRIS_SHAPES: usize = 4;

/// How far a rock's vertices are pulled in from the sphere they start on.
const DEBRIS_JITTER: f32 = 0.35;

/// Mass per cubic world unit.
const DEBRIS_DENSITY: f32 = 3.0;

/// Speed the rock is thrown at when it breaks loose, in world units per second.
const DEBRIS_POP_SPEED: f32 = 1.2;

/// Seconds between checks for rocks that have come to rest somewhere new.
const DEBRIS_SAVE_INTERVAL: f32 = 2.0;

const DEBRIS_SAVE_PATH: &str = "data/terrain/ore_debris.json";

#[derive(Resource)]
pub struct OreDebrisAssets {
    shapes: Vec<Handle<Mesh>>,
    material: Handle<StandardMaterial>,
}

/// Ore that has been dug out but not yet thrown as a rock.
#[derive(Resource, Default)]
pub struct OreDebrisBank {
    volume: f32,
    weighted_center: Vec3,
    spawned: u32,
}

impl OreDebrisBank {
    pub fn deposit(&mut self, volume: f32, center: Vec3) {
        self.volume += volume;
        self.weighted_center += center * volume;
    }

    fn withdraw(&mut self) -> Option<(f32, Vec3)> {
        if self.volume < MIN_DEBRIS_VOLUME {
            return None;
        }
        let center = self.weighted_center / self.volume;
        let volume = self.volume;
        self.volume = 0.0;
        self.weighted_center = Vec3::ZERO;
        self.spawned = self.spawned.wrapping_add(1);
        Some((volume, center))
    }
}

/// A rock lying in the world. `volume` and `shape` are what it takes to rebuild
/// it in the next session; its transform comes from the entity.
#[derive(Component)]
pub struct OreDebris {
    volume: f32,
    shape: usize,
}

/// One rock as it sits on disk. Plain arrays because the bevy math types are
/// only serializable with a feature this build does not enable.
#[derive(Serialize, Deserialize, PartialEq, Clone)]
struct SavedOreDebris {
    position: [f32; 3],
    rotation: [f32; 4],
    volume: f32,
    shape: usize,
}

/// The last thing written to disk, so a world full of settled rocks is not
/// rewritten every interval.
#[derive(Resource, Default)]
pub struct OreDebrisSaveState {
    written: Vec<SavedOreDebris>,
    seconds_since_save: f32,
}

#[inline(always)]
fn hash_unit(seed: u32, index: u32) -> f32 {
    let mut hash = seed.wrapping_mul(0x9E37_79B9) ^ index.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2545_F491);
    hash ^= hash >> 13;
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// A unit-radius lump: an icosphere with each vertex pulled in by its own
/// amount, then faceted so it catches the light like broken rock.
fn rock_mesh(seed: u32) -> Mesh {
    let mut mesh = Sphere::new(1.0).mesh().ico(1).unwrap();
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for (index, position) in positions.iter_mut().enumerate() {
            let pull = 1.0 - DEBRIS_JITTER * hash_unit(seed, index as u32);
            position[0] *= pull;
            position[1] *= pull;
            position[2] *= pull;
        }
    }
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();
    mesh
}

fn debris_radius(volume: f32) -> f32 {
    (volume * 3.0 / (4.0 * std::f32::consts::PI)).cbrt()
}

/// Spawn one rock. Rocks always start fixed: `thaw_simulated_ore_debris`
/// releases them once the terrain under them is simulated, which is what keeps
/// a rock loaded from disk from falling through a world that has not streamed
/// in yet.
fn spawn_rock(
    commands: &mut Commands,
    assets: &OreDebrisAssets,
    transform: Transform,
    velocity: Velocity,
    volume: f32,
    shape: usize,
) {
    let radius = debris_radius(volume);
    commands
        .spawn((
            RigidBody::Fixed,
            Collider::ball(radius * 0.85),
            ColliderMassProperties::Mass(volume * DEBRIS_DENSITY),
            Damping {
                linear_damping: 0.2,
                angular_damping: 0.5,
            },
            velocity,
            transform,
            Visibility::default(),
            OreDebris { volume, shape },
        ))
        .with_children(|children| {
            children.spawn((
                Mesh3d(assets.shapes[shape % DEBRIS_SHAPES].clone()),
                MeshMaterial3d(assets.material.clone()),
                Transform::from_scale(Vec3::splat(radius)),
            ));
        });
}

fn read_saved_debris() -> Vec<SavedOreDebris> {
    read_to_string(get_project_root().join(DEBRIS_SAVE_PATH))
        .ok()
        .and_then(|text| from_str(&text).ok())
        .unwrap_or_default()
}

fn write_saved_debris(rocks: &[SavedOreDebris]) {
    let path = get_project_root().join(DEBRIS_SAVE_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    if let Ok(json) = to_string(rocks) {
        let _ = write(path, json);
    }
}

pub fn setup_ore_debris(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let texture: Handle<Image> = asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::Descriptor(
                SamplerDescriptor {
                    address_mode_u: AddressMode::Repeat,
                    address_mode_v: AddressMode::Repeat,
                    ..Default::default()
                }
                .into(),
            );
        })
        .load(get_project_root().join("assets/source_tiles/ore.png"));
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(texture),
        perceptual_roughness: 0.45,
        metallic: 0.5,
        ..default()
    });
    let shapes = (0..DEBRIS_SHAPES as u32)
        .map(|seed| meshes.add(rock_mesh(seed.wrapping_mul(0x7FEB_352D) | 1)))
        .collect();
    let assets = OreDebrisAssets { shapes, material };
    let saved = read_saved_debris();
    for rock in &saved {
        spawn_rock(
            &mut commands,
            &assets,
            Transform {
                translation: Vec3::from_array(rock.position),
                rotation: Quat::from_array(rock.rotation),
                scale: Vec3::ONE,
            },
            Velocity::zero(),
            rock.volume,
            rock.shape,
        );
    }
    commands.insert_resource(OreDebrisSaveState {
        written: saved,
        seconds_since_save: 0.0,
    });
    commands.insert_resource(assets);
}

/// Throw a rock once enough ore has been dug loose to make one worth having.
pub fn spawn_banked_ore_debris(
    mut commands: Commands,
    mut bank: ResMut<OreDebrisBank>,
    assets: Res<OreDebrisAssets>,
) {
    let Some((volume, center)) = bank.withdraw() else {
        return;
    };
    let seed = bank.spawned;
    let pop = Vec3::new(
        hash_unit(seed, 0) - 0.5,
        hash_unit(seed, 1) * 0.5 + 0.25,
        hash_unit(seed, 2) - 0.5,
    )
    .normalize_or_zero()
        * DEBRIS_POP_SPEED;
    let spin = Vec3::new(
        hash_unit(seed, 3) - 0.5,
        hash_unit(seed, 4) - 0.5,
        hash_unit(seed, 5) - 0.5,
    ) * 6.0;
    spawn_rock(
        &mut commands,
        &assets,
        Transform::from_translation(center),
        Velocity {
            linear: pop,
            angular: spin,
        },
        volume,
        seed as usize,
    );
}

/// Whether the terrain sample at `world_pos` is solid, read from the chunk map
/// rather than from colliders — which is what makes it answerable in the
/// frames before a streamed-in chunk's collider reaches the physics world.
fn sample_is_solid(chunk: &TerrainChunk, world_pos: Vec3, chunk_coord: (i16, i16, i16)) -> bool {
    let (x, y, z) = world_pos_to_voxel_index(&world_pos, &chunk_coord);
    let last = (SAMPLES_PER_CHUNK_DIM_PADDED - 1) as u32;
    chunk.is_solid((x + 1).min(last), (y + 1).min(last), (z + 1).min(last))
}

/// A rock is dynamic only while whatever holds it up is really in the physics
/// world. Terrain colliders exist only inside the simulation radius, and a
/// chunk joins the chunk map several frames before its collider is built, so
/// membership alone would drop a rock through ground that has not arrived yet.
/// Solid ground under a rock therefore has to have its collider; open air under
/// it means there is nothing to wait for and the rock may fall.
pub fn thaw_simulated_ore_debris(
    terrain_chunk_map: Res<TerrainChunkMap>,
    chunk_entity_map: Res<ChunkEntityMap>,
    chunk_colliders: Query<(), (With<ChunkTag>, With<Collider>)>,
    mut debris: Query<(&Transform, &OreDebris, &mut RigidBody)>,
) {
    let map = terrain_chunk_map.0.lock().unwrap();
    for (transform, rock, mut body) in debris.iter_mut() {
        let ground = transform.translation
            - Vec3::Y * (debris_radius(rock.volume) + VOXEL_WORLD_SIZE);
        let ground_coord = world_pos_to_chunk_coord(&ground);
        let simulated = map.contains_key(&world_pos_to_chunk_coord(&transform.translation))
            && map.get(&ground_coord).is_some_and(|chunk| {
                !sample_is_solid(chunk, ground, ground_coord)
                    || chunk_entity_map
                        .get_option(ground_coord)
                        .is_some_and(|(entity, _)| chunk_colliders.contains(*entity))
            });
        let wanted = if simulated {
            RigidBody::Dynamic
        } else {
            RigidBody::Fixed
        };
        if *body != wanted {
            *body = wanted;
        }
    }
}

/// Rocks outlive the chunk they were dug from, so where they came to rest has
/// to survive the session too. Rapier puts a settled rock to sleep and stops
/// touching its transform, so once everything is still this finds nothing to
/// write.
pub fn save_ore_debris(
    time: Res<Time>,
    mut state: ResMut<OreDebrisSaveState>,
    debris: Query<(&Transform, &OreDebris)>,
) {
    state.seconds_since_save += time.delta_secs();
    if state.seconds_since_save < DEBRIS_SAVE_INTERVAL {
        return;
    }
    state.seconds_since_save = 0.0;
    let current: Vec<SavedOreDebris> = debris
        .iter()
        .map(|(transform, rock)| SavedOreDebris {
            position: transform.translation.to_array(),
            rotation: transform.rotation.to_array(),
            volume: rock.volume,
            shape: rock.shape,
        })
        .collect();
    if current == state.written {
        return;
    }
    write_saved_debris(&current);
    state.written = current;
}
