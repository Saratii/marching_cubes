use std::fs::{create_dir_all, read_to_string, write};

use bevy::prelude::*;

use crate::deformable_terrain::{
    file_loader::get_project_root,
    ore_debris::{OreDebris, debris_mass},
};
use crate::elevator::ELEVATOR_TOP_Y;
use crate::player::tools::HeldOre;

const BANK_SAVE_PATH: &str = "data/delivered_copper.txt";

/// Kilograms of copper ore the elevator has hauled out of the mine, totalled
/// over every session.
#[derive(Resource, Default)]
pub struct DeliveredCopper {
    pub kilograms: f32,
}

pub fn load_delivered_copper() -> DeliveredCopper {
    let kilograms = read_to_string(get_project_root().join(BANK_SAVE_PATH))
        .ok()
        .and_then(|text| text.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    DeliveredCopper { kilograms }
}

/// A rock riding a platform past the top of the shaft has left the mine: its
/// mass joins the bank and the rock leaves the world. `save_ore_debris` writes
/// whatever rocks still exist, so this drops it from the debris file too.
pub fn deliver_ore_at_shaft_top(
    mut commands: Commands,
    mut delivered: ResMut<DeliveredCopper>,
    rocks: Query<(Entity, &GlobalTransform, &OreDebris)>,
) {
    for (entity, transform, rock) in rocks.iter() {
        if transform.translation().y < ELEVATOR_TOP_Y {
            continue;
        }
        delivered.kilograms += debris_mass(rock.volume);
        commands.entity(entity).despawn();
    }
}

/// The rock in his hands when the sun caught him goes up with the body, so it
/// is banked rather than lost with the rest of him.
pub fn sell_held_ore_on_death(
    mut commands: Commands,
    mut delivered: ResMut<DeliveredCopper>,
    held: Query<(Entity, &OreDebris), With<HeldOre>>,
) {
    for (entity, rock) in held.iter() {
        delivered.kilograms += debris_mass(rock.volume);
        commands.entity(entity).despawn();
    }
}

pub fn save_delivered_copper(delivered: Res<DeliveredCopper>) {
    if delivered.is_added() || !delivered.is_changed() {
        return;
    }
    let path = get_project_root().join(BANK_SAVE_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    let _ = write(path, delivered.kilograms.to_string());
}
