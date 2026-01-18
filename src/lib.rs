#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod conversions;
pub mod player;
pub mod terrain;
pub mod data_loader;
pub mod sparse_voxel_octree;
pub mod ui;
pub mod marching_cubes;
pub mod settings;
pub mod lighting;