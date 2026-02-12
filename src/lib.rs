#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod constants;
pub mod conversions;
pub mod data_loader;
pub mod lighting;
pub mod marching_cubes;
pub mod player;
pub mod settings;
pub mod sparse_voxel_octree;
pub mod terrain;
pub mod ui;
