#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod constants;
pub mod conversions;
pub mod deformable_terrain;
pub mod lighting;
pub mod player;
pub mod settings;
pub mod terrain;
pub mod ui;
