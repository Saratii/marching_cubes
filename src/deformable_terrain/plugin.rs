use bevy::app::{App, Plugin, Startup, Update};

use crate::deformable_terrain::{driver::chunk_spawn_reciever, file_loader::setup_chunk_loading};

pub struct DeformableTerrainPlugin;

impl Plugin for DeformableTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_chunk_loading)
            .add_systems(Update, chunk_spawn_reciever);
    }
}
