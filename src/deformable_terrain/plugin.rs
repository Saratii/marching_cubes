use std::sync::{Arc, Mutex, atomic::Ordering};

use bevy::prelude::*;
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use serde::{Deserialize, Serialize};

use crate::{
    constants::{NOISE_AMPLITUDE, NOISE_FREQUENCY, WORLD_SEED},
    deformable_terrain::{
        chunk_generator::get_fbm,
        digging::deformation_message_reader,
        driver::{
            Lods, RENDER_RADIUS_SQUARED, chunk_spawn_reciever, info_print, setup_chunk_driver,
        },
        file_loader::setup_chunk_loading,
        terrain::setup_map,
    },
};

#[derive(Clone)]
pub struct NoiseHeightConfig {
    pub generator: GeneratorWrapper<SafeNode>,
    pub frequency: f32,
    pub amplitude: f32,
    pub seed: i32,
}

impl Default for NoiseHeightConfig {
    fn default() -> Self {
        Self {
            generator: get_fbm(),
            frequency: NOISE_FREQUENCY,
            amplitude: NOISE_AMPLITUDE,
            seed: WORLD_SEED,
        }
    }
}

//how the base (unmodified) terrain height is generated
#[derive(Clone)]
pub enum HeightSource {
    Noise(NoiseHeightConfig),
    Flat(f32),
}

impl HeightSource {
    //cheap single-point query of the unmodified terrain height at a world position
    //matches the base surface used by chunk generation (before any digging/deformation)
    pub fn height_at(&self, world_x: f32, world_z: f32) -> f32 {
        match self {
            HeightSource::Noise(config) => {
                config.generator.gen_single_2d(
                    world_x * config.frequency,
                    world_z * config.frequency,
                    config.seed,
                ) * config.amplitude
            }
            HeightSource::Flat(height) => *height,
        }
    }
}

//persistent handle to the base height function shared with the chunk loader threads
//use TerrainHeightSource.0.height_at(x, z) for cheap point queries (e.g. spawn heights)
#[derive(Resource)]
pub struct TerrainHeightSource(pub HeightSource);

#[derive(Component)]
pub struct ChunkTag;

#[derive(Message, Copy, Clone)]
pub enum Deformation {
    Sphere {
        center: Vec3,
        radius: f32,
        strength: f32,
    },
    SphereCarve {
        center: Vec3,
        radius: f32,
    },
}

#[repr(u8)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
pub enum Uniformity {
    NonUniform,
    Dirt,
    Air,
    Unknown,
}

#[derive(Resource)]
pub struct DeformableTerrainConfig {
    pub lods: bool,
}

impl DeformableTerrainConfig {
    pub fn render_radius() -> u32 {
        RENDER_RADIUS_SQUARED.load(Ordering::Relaxed)
    }

    pub fn set_render_radius(radius: u32) {
        RENDER_RADIUS_SQUARED.store(radius, Ordering::Relaxed);
    }

    pub fn default() -> Self {
        DeformableTerrainConfig { lods: false }
    }
}

#[derive(Resource)]
pub struct MoveableCenter {
    pub(crate) center_mutex: Arc<Mutex<Vec3>>,
    last_center: Vec3,
}

impl MoveableCenter {
    pub fn update(&mut self, new_position: Vec3) {
        *(self.center_mutex.lock().unwrap()) = new_position;
        self.last_center = new_position
    }

    pub fn read(&self) -> Vec3 {
        self.last_center
    }
}

pub struct DeformableTerrainPlugin {
    pub lods: bool,
    pub height_source: HeightSource,
}

impl Plugin for DeformableTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MoveableCenter {
            center_mutex: Arc::new(Mutex::new(Vec3::ZERO)),
            last_center: Vec3::ZERO,
        })
        .insert_resource(DeformableTerrainConfig::default())
        .insert_resource(Lods(self.lods))
        .insert_resource(TerrainHeightSource(self.height_source.clone()))
        .add_message::<Deformation>()
        .add_systems(
            Startup,
            (
                info_print,
                setup_chunk_loading,
                setup_chunk_driver,
                setup_map,
            ),
        )
        .add_systems(
            Update,
            (chunk_spawn_reciever, deformation_message_reader).chain(),
        );
    }
}
