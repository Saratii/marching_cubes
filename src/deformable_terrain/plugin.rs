use std::sync::{Arc, Mutex, atomic::Ordering};

use bevy::prelude::*;
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use serde::{Deserialize, Serialize};

use crate::deformable_terrain::{
    digging::deformation_message_reader,
    driver::{Lods, RENDER_RADIUS_SQUARED, chunk_spawn_reciever, info_print, setup_chunk_driver},
    file_loader::setup_chunk_loading,
    terrain::setup_map,
};

#[derive(Resource)]
pub(crate) struct FlatTerrainHeight(pub(crate) Option<f32>);

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

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
    pub flat_terrain_height: Option<f32>,
}

impl Plugin for DeformableTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MoveableCenter {
            center_mutex: Arc::new(Mutex::new(Vec3::ZERO)),
            last_center: Vec3::ZERO,
        })
        .insert_resource(DeformableTerrainConfig::default())
        .insert_resource(Lods(self.lods))
        .insert_resource(FlatTerrainHeight(self.flat_terrain_height))
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
