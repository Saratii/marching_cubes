use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bevy::asset::UnapprovedPathMode;
use bevy::dev_tools::diagnostics_overlay::{DiagnosticsOverlay, DiagnosticsOverlayPlugin};
use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::image::ImageSamplerDescriptor;
use bevy::pbr::diagnostic::MaterialAllocatorDiagnosticPlugin;
use bevy::pbr::{ExtendedMaterial, PbrPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::MeshAllocatorDiagnosticPlugin;
use bevy::window::{PresentMode, WindowMode};
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_rapier3d::plugin::{NoUserData, PhysicsSet, RapierPhysicsPlugin};
// use bevy_rapier3d::render::RapierDebugRenderPlugin;
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;

use marching_cubes::build_initial_area::build_initial_area;
#[cfg(feature = "debug")]
use marching_cubes::deformable_terrain::debug_lines::{
    draw_cluster_debug, draw_collider_debug, draw_lod_debug, draw_voxel_surface_debug,
};
use marching_cubes::deformable_terrain::digging::handle_digging_input;
use marching_cubes::deformable_terrain::driver::{
    FrameStart, INITIAL_CHUNKS_LOADED, record_frame_start,
};
#[cfg(feature = "debug")]
use marching_cubes::deformable_terrain::driver_debug_ui::{spawn_debug_texts, update_debug_texts};
use marching_cubes::deformable_terrain::file_loader::setup_chunk_loading;
use marching_cubes::deformable_terrain::plugin::{
    DeformableTerrainConfig, DeformableTerrainPlugin, HeightSource,
};
use marching_cubes::deformable_terrain::terrain_material::TerrainMaterialExtension;
use marching_cubes::lighting::lighting_main::{
    apply_settings_changes, setup_camera, setup_lighting,
};
use marching_cubes::player::player::{
    CameraController, KeyBindings, camera_look, camera_zoom, free_cam_movement, grab_on_click,
    handle_focus_change, initial_grab_cursor, player_movement, spawn_free_cam_root, spawn_player,
    sync_player_rotation, sync_terrain_center, toggle_first_person, toggle_fly_mode,
    toggle_free_cam, validate_player_spawn,
};
use marching_cubes::settings::settings_driver::{load_settings, save_monitor_on_move};
use marching_cubes::ui::configurable_settings::{
    FpsLimit, MenuFocus, MenuTab, load_configurable_settings,
};
use marching_cubes::ui::crosshair::spawn_crosshair;
use marching_cubes::ui::menu::{SettingsState, menu_toggle, menu_update};

fn main() {
    let settings = load_settings(); //automatically saved state
    let configurable_settings = load_configurable_settings(); //user saved state
    DeformableTerrainConfig::set_render_radius(
        configurable_settings.render_radius_squared.0.to_bits(),
    );
    let window_centered_position = settings.window_centered_position;
    let update_mode = match configurable_settings.fps_limit {
        FpsLimit::Fps60 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 60.0)),
        FpsLimit::Fps120 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 120.0)),
        FpsLimit::Unlimited => UpdateMode::Continuous,
    };
    App::new()
        .insert_resource(settings)
        .insert_resource(SettingsState {
            current_tab: MenuTab::General,
            current_focus: MenuFocus::Tabs,
        })
        .insert_resource(FrameStart(Instant::now()))
        .insert_resource(configurable_settings)
        .insert_resource(KeyBindings::default())
        .insert_resource(CameraController::default())
        .insert_resource(WinitSettings {
            focused_mode: update_mode,
            unfocused_mode: update_mode,
        })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                        position: window_centered_position
                            .map(WindowPosition::At)
                            .unwrap_or(WindowPosition::Automatic),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin {
                    default_sampler: ImageSamplerDescriptor {
                        anisotropy_clamp: 16,
                        ..ImageSamplerDescriptor::linear()
                    },
                })
                .set(PbrPlugin { ..default() })
                .set(AssetPlugin {
                    unapproved_path_mode: UnapprovedPathMode::Allow,
                    ..default()
                }),
            EntityCountDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            DiagnosticsOverlayPlugin,
            MeshAllocatorDiagnosticPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            MaterialAllocatorDiagnosticPlugin::<StandardMaterial>::default(),
            DeformableTerrainPlugin {
                lods: false,
                height_source: HeightSource::Flat(1.0),
                // height_source: HeightSource::Noise(NoiseHeightConfig::default()),
            },
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>::default(
            ),
            // LogDiagnosticsPlugin::default(),
            // RapierDebugRenderPlugin::default(),
        ))
        .add_plugins(PerfUiPlugin)
        // .add_plugins(RenderDiagnosticsPlugin::default())
        .add_systems(
            Startup,
            (
                setup,
                spawn_crosshair,
                spawn_player.after(setup_chunk_loading).after(setup_camera),
                // spawn_minimap.after(spawn_player),
                initial_grab_cursor,
                setup_lighting,
                setup_camera,
                spawn_free_cam_root,
                #[cfg(feature = "debug")]
                spawn_debug_texts,
            ),
        )
        .add_systems(First, record_frame_start)
        .add_systems(
            Update,
            (
                build_initial_area,
                handle_digging_input,
                toggle_first_person,
                camera_zoom,
                camera_look,
                player_movement,
                sync_terrain_center.after(player_movement),
                validate_player_spawn
                    .after(PhysicsSet::SyncBackend)
                    .run_if(|| !INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed)),
                save_monitor_on_move,
                #[cfg(feature = "debug")]
                draw_cluster_debug,
                #[cfg(feature = "debug")]
                draw_collider_debug,
                #[cfg(feature = "debug")]
                draw_lod_debug,
                #[cfg(feature = "debug")]
                draw_voxel_surface_debug,
                menu_toggle,
                menu_update.after(menu_toggle),
                handle_focus_change,
                grab_on_click,
                toggle_fly_mode,
                apply_settings_changes,
            ),
        )
        .add_systems(
            Update,
            (
                toggle_free_cam,
                free_cam_movement,
                sync_player_rotation,
                #[cfg(feature = "debug")]
                update_debug_texts,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(DiagnosticsOverlay::mesh_and_standard_material());
    commands.spawn(PerfUiDefaultEntries::default());
}
