use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string_pretty};
use std::fs::{create_dir_all, read_to_string, write};
use std::path::PathBuf;

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
    2200.0 * 2200.0,
    2400.0 * 2400.0,
    2600.0 * 2600.0,
    2800.0 * 2800.0,
    3000.0 * 3000.0,
];
const _: () = assert!(RENDER_RADIUS_STEPS[0] as u64 >= SIMULATION_RADIUS as u64);
pub const DEFAULT_RENDER_RADIUS_SQUARED: f32 = 1000.0 * 1000.0;
const DEFAULT_DIG_RADIUS: f32 = 2.0;
// dig_strength is world units/second the dug surface advances at the brush
const DEFAULT_DIG_STRENGTH: f32 = 3.0;
const DIG_RADIUS_STEP: f32 = 1.0;
const DIG_RADIUS_RANGE: (f32, f32) = (1.0, 40.0);
const DIG_STRENGTH_STEP: f32 = 0.25;
const DIG_STRENGTH_RANGE: (f32, f32) = (0.25, 10.0);
// ambient brightness is in cd/m^2 pre-exposure; at the default ev100 of 5.0 the
// scene is scaled by ~1/38, so values in the tens already read as visible
const DEFAULT_AMBIENT_BRIGHTNESS: f32 = 0.0;
const AMBIENT_BRIGHTNESS_STEP: f32 = 2_500.0;
const AMBIENT_BRIGHTNESS_RANGE: (f32, f32) = (0.0, 50_000.0);
const DEFAULT_SUN_ILLUMINANCE: f32 = 0.0;
const SUN_ILLUMINANCE_STEP: f32 = 5_000.0;
const SUN_ILLUMINANCE_RANGE: (f32, f32) = (0.0, 150_000.0);
pub const DEFAULT_LANTERN_BRIGHTNESS: f32 = 500_000.0;
const LANTERN_BRIGHTNESS_RANGE: (f32, f32) = (10_000.0, 20_000_000.0);
// headlamp brightness is the spotlight's luminous power in lumens
const DEFAULT_HEADLAMP_BRIGHTNESS: f32 = 327_680.0;
const HEADLAMP_BRIGHTNESS_RANGE: (f32, f32) = (100_000.0, 200_000_000.0);
// lamp brightness presses scale by this instead of adding, so dim lamps stay
// adjustable; the range's low end is the dimmest lit value and stepping below
// it turns the lamp off
const BRIGHTNESS_STEP_FACTOR: f32 = 1.25;
// god ray brightness is the spotlight's luminous power in lumens
const DEFAULT_GOD_RAY_BRIGHTNESS: f32 = 2_000_000.0;
const GOD_RAY_BRIGHTNESS_STEP: f32 = 1_000_000.0;
const GOD_RAY_BRIGHTNESS_RANGE: (f32, f32) = (0.0, 200_000_000.0);
// lower ev100 means a longer exposure, so a brighter image
const DEFAULT_EXPOSURE_EV100: f32 = 5.0;
const EXPOSURE_EV100_STEP: f32 = 0.25;
const EXPOSURE_EV100_RANGE: (f32, f32) = (5.0, 20.0);

fn default_dig_radius() -> f32 {
    DEFAULT_DIG_RADIUS
}

fn default_dig_strength() -> f32 {
    DEFAULT_DIG_STRENGTH
}

fn default_ambient_brightness() -> f32 {
    DEFAULT_AMBIENT_BRIGHTNESS
}

fn default_sun_illuminance() -> f32 {
    DEFAULT_SUN_ILLUMINANCE
}

fn step_brightness(value: f32, dir_next: bool, range: (f32, f32)) -> f32 {
    let (min, max) = range;
    if dir_next {
        if value < min {
            min
        } else {
            (value * BRIGHTNESS_STEP_FACTOR).min(max)
        }
    } else if value <= min {
        0.0
    } else {
        (value / BRIGHTNESS_STEP_FACTOR).max(min)
    }
}

fn default_lantern_brightness() -> f32 {
    DEFAULT_LANTERN_BRIGHTNESS
}

fn default_exposure_ev100() -> f32 {
    DEFAULT_EXPOSURE_EV100
}

fn default_god_ray_brightness() -> f32 {
    DEFAULT_GOD_RAY_BRIGHTNESS
}

fn default_headlamp_brightness() -> f32 {
    DEFAULT_HEADLAMP_BRIGHTNESS
}

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
    Lighting,
    #[cfg(feature = "debug")]
    Debug,
}

impl MenuTab {
    #[cfg(feature = "debug")]
    pub const ALL: &'static [MenuTab] = &[MenuTab::General, MenuTab::Lighting, MenuTab::Debug];
    #[cfg(not(feature = "debug"))]
    pub const ALL: &'static [MenuTab] = &[MenuTab::General, MenuTab::Lighting];

    pub fn to_display_string(&self) -> &str {
        match self {
            MenuTab::General => "General",
            MenuTab::Lighting => "Lighting",
            #[cfg(feature = "debug")]
            MenuTab::Debug => "Debug",
        }
    }
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
    ShowVoxelsToggle,
    FpsChange,
    ShadowsToggle,
    RenderRadiusChange,
    FogStartMultiplier,
    FogEndMultiplier,
    DistanceFogToggle,
    OcclusionCullingToggle,
    DigRadiusChange,
    DigStrengthChange,
    AmbientBrightnessChange,
    SunIlluminanceChange,
    LanternBrightnessChange,
    GodRayBrightnessChange,
    HeadlampBrightnessChange,
    ExposureChange,
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
            SettingsType::ShowVoxelsToggle => format!("Show Voxels: {}", on_off(s.show_voxels)),
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
            SettingsType::OcclusionCullingToggle => {
                format!("Occlusion Culling: {}", on_off(s.occlusion_culling))
            }
            SettingsType::DigRadiusChange => format!("Dig Radius: {:.0}", s.dig_radius),
            SettingsType::DigStrengthChange => format!("Dig Strength: {:.1} u/s", s.dig_strength),
            SettingsType::AmbientBrightnessChange => {
                format!("Ambient Light: {:.0}", s.ambient_brightness)
            }
            SettingsType::SunIlluminanceChange => {
                format!("Sunlight: {:.0} lx", s.sun_illuminance)
            }
            SettingsType::LanternBrightnessChange => {
                format!("Lantern Brightness: {:.1}M lm", s.lantern_brightness / 1e6)
            }
            SettingsType::GodRayBrightnessChange => {
                format!("God Ray Brightness: {:.0} lm", s.god_ray_brightness)
            }
            SettingsType::HeadlampBrightnessChange => {
                format!("Headlamp Brightness: {:.0} lm", s.headlamp_brightness)
            }
            SettingsType::ExposureChange => {
                format!("Exposure: EV{:.2}", s.exposure_ev100)
            }
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
            SettingsType::ShowVoxelsToggle => settings.show_voxels = !settings.show_voxels,
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
            SettingsType::OcclusionCullingToggle => {
                settings.occlusion_culling = !settings.occlusion_culling
            }
            SettingsType::DigRadiusChange => {
                let step = if dir_next {
                    DIG_RADIUS_STEP
                } else {
                    -DIG_RADIUS_STEP
                };
                settings.dig_radius =
                    (settings.dig_radius + step).clamp(DIG_RADIUS_RANGE.0, DIG_RADIUS_RANGE.1);
            }
            SettingsType::DigStrengthChange => {
                let step = if dir_next {
                    DIG_STRENGTH_STEP
                } else {
                    -DIG_STRENGTH_STEP
                };
                settings.dig_strength = (settings.dig_strength + step)
                    .clamp(DIG_STRENGTH_RANGE.0, DIG_STRENGTH_RANGE.1);
            }
            SettingsType::AmbientBrightnessChange => {
                let step = if dir_next {
                    AMBIENT_BRIGHTNESS_STEP
                } else {
                    -AMBIENT_BRIGHTNESS_STEP
                };
                settings.ambient_brightness = (settings.ambient_brightness + step)
                    .clamp(AMBIENT_BRIGHTNESS_RANGE.0, AMBIENT_BRIGHTNESS_RANGE.1);
            }
            SettingsType::SunIlluminanceChange => {
                let step = if dir_next {
                    SUN_ILLUMINANCE_STEP
                } else {
                    -SUN_ILLUMINANCE_STEP
                };
                settings.sun_illuminance = (settings.sun_illuminance + step)
                    .clamp(SUN_ILLUMINANCE_RANGE.0, SUN_ILLUMINANCE_RANGE.1);
            }
            SettingsType::LanternBrightnessChange => {
                settings.lantern_brightness = step_brightness(
                    settings.lantern_brightness,
                    dir_next,
                    LANTERN_BRIGHTNESS_RANGE,
                );
            }
            SettingsType::GodRayBrightnessChange => {
                let step = if dir_next {
                    GOD_RAY_BRIGHTNESS_STEP
                } else {
                    -GOD_RAY_BRIGHTNESS_STEP
                };
                settings.god_ray_brightness = (settings.god_ray_brightness + step)
                    .clamp(GOD_RAY_BRIGHTNESS_RANGE.0, GOD_RAY_BRIGHTNESS_RANGE.1);
            }
            SettingsType::HeadlampBrightnessChange => {
                settings.headlamp_brightness = step_brightness(
                    settings.headlamp_brightness,
                    dir_next,
                    HEADLAMP_BRIGHTNESS_RANGE,
                );
            }
            // right brightens, so it steps ev100 down
            SettingsType::ExposureChange => {
                let step = if dir_next {
                    -EXPOSURE_EV100_STEP
                } else {
                    EXPOSURE_EV100_STEP
                };
                settings.exposure_ev100 = (settings.exposure_ev100 + step)
                    .clamp(EXPOSURE_EV100_RANGE.0, EXPOSURE_EV100_RANGE.1);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Resource, Debug)]
pub struct ConfigurableSettings {
    pub show_chunks: bool,
    pub show_voxels: bool,
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
    #[serde(default = "default_dig_radius")]
    pub dig_radius: f32,
    #[serde(default = "default_dig_strength")]
    pub dig_strength: f32,
    #[serde(default = "default_ambient_brightness")]
    pub ambient_brightness: f32,
    #[serde(default = "default_sun_illuminance")]
    pub sun_illuminance: f32,
    #[serde(default = "default_lantern_brightness")]
    pub lantern_brightness: f32,
    #[serde(default = "default_exposure_ev100")]
    pub exposure_ev100: f32,
    #[serde(default = "default_god_ray_brightness")]
    pub god_ray_brightness: f32,
    #[serde(default = "default_headlamp_brightness")]
    pub headlamp_brightness: f32,
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
            show_voxels: false,
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
            dig_radius: DEFAULT_DIG_RADIUS,
            dig_strength: DEFAULT_DIG_STRENGTH,
            ambient_brightness: DEFAULT_AMBIENT_BRIGHTNESS,
            sun_illuminance: DEFAULT_SUN_ILLUMINANCE,
            lantern_brightness: DEFAULT_LANTERN_BRIGHTNESS,
            exposure_ev100: DEFAULT_EXPOSURE_EV100,
            god_ray_brightness: DEFAULT_GOD_RAY_BRIGHTNESS,
            headlamp_brightness: DEFAULT_HEADLAMP_BRIGHTNESS,
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
