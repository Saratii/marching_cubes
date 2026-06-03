use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string_pretty};
use std::fs::{create_dir_all, read_to_string, write};
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;

use crate::constants::SIMULATION_RADIUS;

const CONFIG_PATH: &str = "data/configurable_settings.json";
const RENDER_RADIUS_STEPS: &[f32] = &[
    200.0 * 200.0,
    400.0 * 400.0,
    600.0 * 600.0,
    800.0 * 800.0,
    1000.0 * 1000.0,
    1200.0 * 1200.0,
    1400.0 * 1400.0,
    1600.0 * 1600.0,
    1800.0 * 1800.0,
    2000.0 * 2000.0,
];
const _: () = assert!(RENDER_RADIUS_STEPS[0] as u64 >= SIMULATION_RADIUS as u64);
const DEFAULT_RENDER_RADIUS_SQUARED: f32 = 1000.0 * 1000.0;

pub static RENDER_RADIUS_SQUARED: AtomicU32 = AtomicU32::new(DEFAULT_RENDER_RADIUS_SQUARED as u32);

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderRadiusSquared(pub f32);

impl RenderRadiusSquared {
    pub fn next_step(&self) -> Self {
        let pos = RENDER_RADIUS_STEPS
            .iter()
            .position(|&v| v == self.0)
            .unwrap_or(0);
        RenderRadiusSquared(RENDER_RADIUS_STEPS[(pos + 1).min(RENDER_RADIUS_STEPS.len() - 1)])
    }

    pub fn prev_step(&self) -> Self {
        let pos = RENDER_RADIUS_STEPS
            .iter()
            .position(|&v| v == self.0)
            .unwrap_or(0);
        RenderRadiusSquared(RENDER_RADIUS_STEPS[pos.saturating_sub(1)])
    }

    pub fn to_display_string(&self) -> String {
        format!("{}", (self.0 as u32).isqrt())
    }
}

impl Default for RenderRadiusSquared {
    fn default() -> Self {
        RenderRadiusSquared(DEFAULT_RENDER_RADIUS_SQUARED)
    }
}

#[derive(Serialize, Deserialize, Resource, Debug, Clone, Copy, PartialEq)]
pub enum FpsLimit {
    Fps60,
    Fps120,
    Unlimited,
}

impl FpsLimit {
    pub fn next(&self) -> Self {
        match self {
            FpsLimit::Fps60 => FpsLimit::Fps120,
            FpsLimit::Fps120 => FpsLimit::Unlimited,
            FpsLimit::Unlimited => FpsLimit::Fps60,
        }
    }

    pub fn previous(&self) -> Self {
        match self {
            FpsLimit::Fps60 => FpsLimit::Unlimited,
            FpsLimit::Fps120 => FpsLimit::Fps60,
            FpsLimit::Unlimited => FpsLimit::Fps120,
        }
    }

    pub fn to_display_string(&self) -> &str {
        match self {
            FpsLimit::Fps60 => "60",
            FpsLimit::Fps120 => "120",
            FpsLimit::Unlimited => "Unlimited",
        }
    }
}

impl Default for FpsLimit {
    fn default() -> Self {
        FpsLimit::Fps60
    }
}

#[derive(Serialize, Deserialize, Resource, Debug, Clone, Copy, PartialEq)]
pub enum MenuTab {
    General,
    #[cfg(feature = "debug")]
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MenuFocus {
    Tabs,
    Setting(usize),
}

#[derive(Copy, PartialEq, Clone)]
pub enum SettingsType {
    Lod1Toggle,
    Lod2Toggle,
    Lod3Toggle,
    Lod4Toggle,
    Lod5Toggle,
    ShowChunksToggle,
    FpsChange,
    ShadowsToggle,
    RenderRadiusChange,
    FogStartMultiplier,
    FogEndMultiplier,
    DistanceFogToggle,
    OcclusionCullingToggle,
}

impl SettingsType {
    pub fn text(&self, s: &ConfigurableSettings) -> String {
        const fn on_off(b: bool) -> &'static str {
            if b { "ON" } else { "OFF" }
        }
        match self {
            SettingsType::Lod1Toggle => format!("LOD 1: {}", on_off(s.debug_lod_1)),
            SettingsType::Lod2Toggle => format!("LOD 2: {}", on_off(s.debug_lod_2)),
            SettingsType::Lod3Toggle => format!("LOD 3: {}", on_off(s.debug_lod_3)),
            SettingsType::Lod4Toggle => format!("LOD 4: {}", on_off(s.debug_lod_4)),
            SettingsType::Lod5Toggle => format!("LOD 5: {}", on_off(s.debug_lod_5)),
            SettingsType::ShowChunksToggle => format!("Show Chunks: {}", on_off(s.show_chunks)),
            SettingsType::FpsChange => format!("FPS Limit: {}", s.fps_limit.to_display_string()),
            SettingsType::ShadowsToggle => format!("Shadows: {}", on_off(s.shadows)),
            SettingsType::RenderRadiusChange => format!(
                "Render Radius: {}",
                s.render_radius_squared.to_display_string()
            ),
            SettingsType::FogStartMultiplier => {
                format!("Fog Start Multiplier: {:.2}", s.fog_start_multiplier)
            }
            SettingsType::FogEndMultiplier => {
                format!("Fog End Multiplier: {:.2}", s.fog_end_multiplier)
            }
            SettingsType::DistanceFogToggle => format!("Distance Fog: {}", on_off(s.distance_fog)),
            SettingsType::OcclusionCullingToggle => format!("Occlusion Culling: {}", on_off(s.occlusion_culling)),
        }
    }

    pub fn cycle(&self, settings: &mut ConfigurableSettings, dir_next: bool) {
        match self {
            SettingsType::FpsChange => {
                settings.fps_limit = if dir_next {
                    settings.fps_limit.next()
                } else {
                    settings.fps_limit.previous()
                };
            }
            SettingsType::Lod1Toggle => settings.debug_lod_1 = !settings.debug_lod_1,
            SettingsType::Lod2Toggle => settings.debug_lod_2 = !settings.debug_lod_2,
            SettingsType::Lod3Toggle => settings.debug_lod_3 = !settings.debug_lod_3,
            SettingsType::Lod4Toggle => settings.debug_lod_4 = !settings.debug_lod_4,
            SettingsType::Lod5Toggle => settings.debug_lod_5 = !settings.debug_lod_5,
            SettingsType::ShowChunksToggle => settings.show_chunks = !settings.show_chunks,
            SettingsType::ShadowsToggle => settings.shadows = !settings.shadows,
            SettingsType::RenderRadiusChange => {
                settings.render_radius_squared = if dir_next {
                    settings.render_radius_squared.next_step()
                } else {
                    settings.render_radius_squared.prev_step()
                };
            }
            SettingsType::FogStartMultiplier => {
                let new = settings.fog_start_multiplier + if dir_next { 0.05 } else { -0.05 };
                let new = new.clamp(0.0, settings.fog_end_multiplier - 0.05);
                settings.fog_start_multiplier = new;
            }
            SettingsType::FogEndMultiplier => {
                let new = settings.fog_end_multiplier + if dir_next { 0.05 } else { -0.05 };
                let new = new.clamp(settings.fog_start_multiplier + 0.05, 1.0);
                settings.fog_end_multiplier = new;
            }
            SettingsType::DistanceFogToggle => settings.distance_fog = !settings.distance_fog,
            SettingsType::OcclusionCullingToggle => settings.occlusion_culling = !settings.occlusion_culling,
        }
    }
}

#[derive(Serialize, Deserialize, Resource, Debug)]
pub struct ConfigurableSettings {
    pub show_chunks: bool,
    pub fps_limit: FpsLimit,
    pub debug_lod_1: bool,
    pub debug_lod_2: bool,
    pub debug_lod_3: bool,
    pub debug_lod_4: bool,
    pub debug_lod_5: bool,
    pub shadows: bool,
    pub render_radius_squared: RenderRadiusSquared,
    pub fog_start_multiplier: f32,
    pub fog_end_multiplier: f32,
    pub distance_fog: bool,
    pub occlusion_culling: bool,
}

pub fn load_configurable_settings() -> ConfigurableSettings {
    read_to_string(CONFIG_PATH)
        .ok()
        .and_then(|s| from_str(&s).ok())
        .unwrap_or_default()
}

impl Default for ConfigurableSettings {
    fn default() -> Self {
        ConfigurableSettings {
            show_chunks: false,
            fps_limit: FpsLimit::default(),
            debug_lod_1: false,
            debug_lod_2: false,
            debug_lod_3: false,
            debug_lod_4: false,
            debug_lod_5: false,
            shadows: true,
            render_radius_squared: RenderRadiusSquared::default(),
            fog_start_multiplier: 0.7,
            fog_end_multiplier: 0.8,
            distance_fog: true,
            occlusion_culling: true,
        }
    }
}

pub fn save_configurable_settings(settings: &ConfigurableSettings) {
    let path = PathBuf::from(CONFIG_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    if let Ok(json) = to_string_pretty(settings) {
        let _ = write(path, json);
    }
}
