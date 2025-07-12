#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod marching_cubes;
pub mod terrain_generation;
pub mod triangle_table;