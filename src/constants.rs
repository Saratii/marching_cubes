use bevy::math::Vec3;

pub const Z0_RADIUS: f32 = 80.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z1_RADIUS: f32 = 100.0; //in world units. Distance where chunks are loaded at full res but not stored in memory.
pub const Z2_RADIUS: f32 = 2600.0;
pub const CHUNK_WORLD_SIZE: f32 = 12.0; //in world units, required by noise to be an integer and even
pub const SAMPLES_PER_CHUNK_DIM: usize = 64; // Number of voxel sample points
pub const CHUNKS_PER_CLUSTER_DIM: usize = 5; //number of chunks along one edge of a cluster
pub const REDUCED_LOD_1_RADIUS: f32 = 100.0; //red
pub const REDUCED_LOD_2_RADIUS: f32 = 200.0; //cyan
pub const REDUCED_LOD_3_RADIUS: f32 = 550.0; //blue
pub const REDUCED_LOD_4_RADIUS: f32 = 1150.0; //yellow
pub const REDUCED_LOD_5_RADIUS: f32 = 2600.0; //purple
pub const NOISE_SEED: i32 = 111; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.0005; // Frequency of the noise
pub const NOISE_AMPLITUDE: f32 = 300.0; // Amplitude of the noise
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0., 0., 0.);
pub const PLAYER_CUBOID_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);
pub const CAMERA_FIRST_PERSON_OFFSET: Vec3 = Vec3::new(0., 0.75 * PLAYER_CUBOID_SIZE.y, 0.);

pub const CLUSTER_WORLD_LENGTH: f32 = CHUNK_WORLD_SIZE * CHUNKS_PER_CLUSTER_DIM as f32;
pub const SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK_DIM.pow(3);
pub const SAMPLES_PER_CHUNK_2D: usize = SAMPLES_PER_CHUNK_DIM.pow(2);
pub const HALF_CHUNK: f32 = CHUNK_WORLD_SIZE / 2.0;
pub const MAX_RENDER_RADIUS: f32 = Z0_RADIUS.max(Z1_RADIUS).max(Z2_RADIUS);
pub const VOXEL_WORLD_SIZE: f32 = CHUNK_WORLD_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
pub const CHUNKS_PER_CLUSTER: usize = CHUNKS_PER_CLUSTER_DIM.pow(3);
pub const CHUNKS_PER_CLUSTER_2D: usize = CHUNKS_PER_CLUSTER_DIM.pow(2);
pub const MAX_RENDER_RADIUS_SQUARED: f32 = MAX_RENDER_RADIUS * MAX_RENDER_RADIUS;
