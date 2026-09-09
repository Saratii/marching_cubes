use std::time::{Duration, Instant};

use bevy::asset::UnapprovedPathMode;
use bevy::dev_tools::diagnostics_overlay::{DiagnosticsOverlay, DiagnosticsOverlayPlugin};
use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::image::ImageSamplerDescriptor;
use bevy::pbr::PbrPlugin;
use bevy::pbr::diagnostic::MaterialAllocatorDiagnosticPlugin;
use bevy::prelude::*;
use bevy::render::diagnostic::MeshAllocatorDiagnosticPlugin;
use bevy::window::{PresentMode, WindowMode};
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
// use bevy_rapier3d::render::RapierDebugRenderPlugin;
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;

use marching_cubes::build_initial_area::{InitialAreaBuilt, build_initial_area};
#[cfg(feature = "debug")]
use marching_cubes::deformable_terrain::debug_lines::{
    draw_cluster_debug, draw_collider_debug, draw_lod_debug, draw_voxel_surface_debug,
};
use marching_cubes::deformable_terrain::digging::handle_digging_input;
use marching_cubes::deformable_terrain::driver::{FrameStart, record_frame_start};
#[cfg(feature = "debug")]
use marching_cubes::deformable_terrain::driver_debug_ui::{spawn_debug_texts, update_debug_texts};
use marching_cubes::deformable_terrain::plugin::{
    DeformableTerrainConfig, DeformableTerrainPlugin, HeightSource,
};
use marching_cubes::elevator::{setup_elevator, update_elevator};
use marching_cubes::lanterns::spawn_lanterns;
use marching_cubes::lighting::lighting_main::{
    apply_settings_changes, setup_camera, setup_lighting,
};
use marching_cubes::ore_bank::{
    deliver_ore_at_shaft_top, load_delivered_copper, save_delivered_copper, sell_held_ore_on_death,
};
use marching_cubes::player::plugin::PlayerPlugin;
use marching_cubes::player::sun_death::SunDeath;
use marching_cubes::player::tools::{handle_hand_input, select_tool};
use marching_cubes::settings::settings_driver::{load_settings, save_monitor_on_move};
use marching_cubes::ui::configurable_settings::{
    FpsLimit, MenuFocus, MenuTab, load_configurable_settings,
};
use marching_cubes::ui::copper_counter::{spawn_copper_counter, update_copper_counter};
use marching_cubes::ui::crosshair::spawn_crosshair;
use marching_cubes::ui::tool_bar::{spawn_tool_bar, update_tool_bar};
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
        .insert_resource(load_delivered_copper())
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
            PlayerPlugin,
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
                spawn_tool_bar,
                spawn_copper_counter,
                setup_lighting,
                setup_camera,
                setup_elevator,
                #[cfg(feature = "debug")]
                spawn_debug_texts,
            ),
        )
        .add_systems(First, record_frame_start)
        .add_systems(
            Update,
            (
                build_initial_area,
                update_elevator,
                (select_tool, handle_digging_input, handle_hand_input)
                    .chain()
                    .run_if(not(resource_exists::<SunDeath>)),
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
                apply_settings_changes,
            ),
        )
        .add_systems(
            Update,
            (
                spawn_lanterns.run_if(resource_exists::<InitialAreaBuilt>),
                update_tool_bar,
                deliver_ore_at_shaft_top.after(update_elevator),
                sell_held_ore_on_death
                    .run_if(resource_added::<SunDeath>)
                    .after(deliver_ore_at_shaft_top),
                save_delivered_copper.after(sell_held_ore_on_death),
                update_copper_counter.after(sell_held_ore_on_death),
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
