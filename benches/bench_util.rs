use marching_cubes::deformable_terrain::{
    chunk_generator::{
        calculate_chunk_start, chunk_contains_surface, compute_heightmap_gradients,
        generate_chunk_into_buffers, generate_noise_height_samples, generate_terrain_heights,
        get_fbm,
    },
    driver::ChunkBuffers,
};

pub fn find_chunk_with_surface() -> (i16, i16, i16) {
    let mut chunk_buffers = ChunkBuffers::new();
    let chunk_coord = (0, 0, 0);
    let chunk_start = calculate_chunk_start(&chunk_coord);
    let fbm = get_fbm();
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, &fbm);
    generate_terrain_heights(&mut chunk_buffers.heightmap, &noise_samples);
    compute_heightmap_gradients(
        &mut chunk_buffers.dhdx,
        &mut chunk_buffers.dhdz,
        &noise_samples,
    );
    for chunk_y in -100..100 {
        let chunk_coord = (0, chunk_y, 0);
        let chunk_start = calculate_chunk_start(&chunk_coord);
        generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
        if chunk_contains_surface(&chunk_buffers.density) {
            return chunk_coord;
        }
    }
    panic!("No chunk with surface found in the tested range");
}
