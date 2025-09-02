#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod conversions;
pub mod marching_cubes;
pub mod player;
pub mod terrain;
pub mod triangle_table;
pub mod data_loader;
