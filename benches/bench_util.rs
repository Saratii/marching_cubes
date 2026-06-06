use marching_cubes::{
    data_loader::driver::ChunkBuffers,
    terrain::chunk_generator::{
        calculate_chunk_start, chunk_contains_surface, generate_chunk_into_buffers, get_fbm,
    },
};

pub fn find_chunk_with_surface() -> (i16, i16, i16) {
    let mut chunk_buffers = ChunkBuffers::new();
    let fbm = get_fbm();
    for chunk_y in -100..100 {
        let chunk_coord = (0, chunk_y, 0);
        let chunk_start = calculate_chunk_start(&chunk_coord);
        generate_chunk_into_buffers(&fbm, chunk_start, &mut chunk_buffers);
        if chunk_contains_surface(&chunk_buffers.density) {
            return chunk_coord;
        }
    }
    panic!("No chunk with surface found in the tested range");
}
