use std::sync::{Arc, Mutex};

use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::resource::Resource,
    math::Vec3,
};

use crate::deformable_terrain::{driver::chunk_spawn_reciever, file_loader::setup_chunk_loading};

pub struct DeformableTerrainPlugin;

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

impl Plugin for DeformableTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MoveableCenter {
            center_mutex: Arc::new(Mutex::new(Vec3::ZERO)),
            last_center: Vec3::ZERO,
        })
        .add_systems(Startup, setup_chunk_loading)
        .add_systems(Update, chunk_spawn_reciever);
    }
}
