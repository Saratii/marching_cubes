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

#[derive(Serialize, Deserialize, Resource, Debug)]
pub struct ConfigurableSettings {
    pub fps_limit: FpsLimit,
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
            fps_limit: FpsLimit::default(),
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
