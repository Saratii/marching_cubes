use bevy::prelude::*;
use bevy::window::{Monitor, PrimaryWindow, WindowMoved};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string_pretty};
use std::fs::{create_dir_all, read_to_string, write};
use std::path::PathBuf;

const SETTINGS_PATH: &str = "data/settings.json";

#[derive(Serialize, Deserialize, Resource, Debug)]
pub struct Settings {
    //centered monitor position for initial window location
    pub window_centered_position: Option<IVec2>,
}

//load settings from json file, or return default if file not found or invalid
pub fn load_settings() -> Settings {
    read_to_string(SETTINGS_PATH)
        .ok()
        .and_then(|s| from_str(&s).ok())
        .unwrap_or_default()
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            window_centered_position: None,
        }
    }
}

//write settings to json file
fn save_settings(settings: &Settings) {
    let path = PathBuf::from(SETTINGS_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    if let Ok(json) = to_string_pretty(settings) {
        let _ = write(path, json);
    }
}

//after monitor moves, save the new centered position to settings
pub fn save_monitor_on_move(
    mut window_moved: MessageReader<WindowMoved>,
    window: Single<&Window, With<PrimaryWindow>>,
    monitors: Query<&Monitor>,
) {
    if window_moved.read().last().is_none() {
        return;
    }
    let WindowPosition::At(pos) = window.position else {
        return;
    };
    let window_size = window.physical_size().as_ivec2();
    let centered_pos = monitors.iter().find_map(|m| {
        let m_pos = m.physical_position;
        let m_size = m.physical_size().as_ivec2();
        let in_bounds = pos.x >= m_pos.x
            && pos.x < m_pos.x + m_size.x
            && pos.y >= m_pos.y
            && pos.y < m_pos.y + m_size.y;
        if in_bounds {
            Some(m_pos + (m_size - window_size) / 2)
        } else {
            None
        }
    });
    if let Some(centered_pos) = centered_pos {
        save_settings(&Settings {
            window_centered_position: Some(centered_pos),
        });
    }
}
