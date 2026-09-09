use std::sync::atomic::Ordering;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;

use crate::{
    deformable_terrain::{driver::INITIAL_CHUNKS_LOADED, file_loader::setup_chunk_loading},
    elevator::update_elevator,
    lighting::lighting_main::setup_camera,
    player::{
        headlamp::{aim_headlamp, toggle_headlamp},
        player::{
            CameraController, KeyBindings, camera_look, camera_zoom, free_cam_movement,
            grab_on_click, handle_focus_change, initial_grab_cursor, player_movement,
            spawn_free_cam_root, spawn_player, sync_player_rotation, sync_terrain_center,
            toggle_first_person, toggle_fly_mode, toggle_free_cam, update_third_person_camera,
            validate_player_spawn,
        },
        player_visual::animate_player_limbs,
        sun_death::{SunDeath, spawn_sun_flash, trigger_sun_death, update_sun_death},
    },
};

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<KeyBindings>()
            .init_resource::<CameraController>()
            .add_systems(
                Startup,
                (
                    spawn_player.after(setup_chunk_loading).after(setup_camera),
                    initial_grab_cursor,
                    spawn_free_cam_root,
                    spawn_sun_flash,
                ),
            )
            .add_systems(
                Update,
                (
                    (
                        toggle_first_person,
                        camera_zoom,
                        camera_look,
                        update_third_person_camera,
                    )
                        .chain(),
                    player_movement.run_if(not(resource_exists::<SunDeath>)),
                    sync_terrain_center.after(player_movement),
                    validate_player_spawn
                        .after(PhysicsSet::SyncBackend)
                        .run_if(|| !INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed)),
                    sync_player_rotation.after(camera_look),
                    animate_player_limbs.after(player_movement),
                    aim_headlamp.after(camera_look),
                    toggle_headlamp,
                    handle_focus_change,
                    grab_on_click,
                    toggle_fly_mode,
                    toggle_free_cam,
                    free_cam_movement,
                    trigger_sun_death.after(update_elevator),
                    update_sun_death.after(trigger_sun_death),
                ),
            );
    }
}
