use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string_pretty};
use std::fs::{create_dir_all, read_to_string, write};
use std::path::PathBuf;

const CONFIG_PATH: &str = "data/configurable_settings.json";

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
