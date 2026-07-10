use bevy::prelude::*;
use rustc_hash::FxHashMap;

//store mesh handle to be able to replace mesh without the entity being spawned to avoid crash NotSpawned(ValidButNotSpawned(EntityValidButNotSpawnedError
#[derive(Resource)]
pub struct ChunkEntityMap(FxHashMap<(i16, i16, i16), (Entity, Handle<Mesh>)>);

impl ChunkEntityMap {
    pub(crate) fn new() -> ChunkEntityMap {
        ChunkEntityMap(FxHashMap::default())
    }

    pub(crate) fn insert(&mut self, chunk_coord: (i16, i16, i16), entity: (Entity, Handle<Mesh>)) {
        #[cfg(feature = "debug")]
        {
            assert!(
                self.0.insert(chunk_coord, entity).is_none(),
                "ChunkEntityMap::insert: chunk coord {chunk_coord:?} already had an entity"
            );
        }
        #[cfg(not(feature = "debug"))]
        {
            self.0.insert(chunk_coord, entity);
        }
    }

    pub(crate) fn replace_mesh_handle(
        &mut self,
        chunk_coord: (i16, i16, i16),
        new_mesh_handle: Handle<Mesh>,
    ) {
        let (_, mesh_handle) = self.0.get_mut(&chunk_coord).unwrap();
        *mesh_handle = new_mesh_handle;
    }

    pub fn get(&self, chunk_coord: (i16, i16, i16)) -> (Entity, Handle<Mesh>) {
        #[cfg(feature = "debug")]
        {
            let result = self.0.get(&chunk_coord);
            assert!(
                result.is_some(),
                "ChunkEntityMap::get: chunk coord {chunk_coord:?} had no entity"
            );
            result.unwrap().clone()
        }
        #[cfg(not(feature = "debug"))]
        {
            self.0.get(&chunk_coord).unwrap().clone()
        }
    }

    pub fn get_option(&self, chunk_coord: (i16, i16, i16)) -> Option<&(Entity, Handle<Mesh>)> {
        self.0.get(&chunk_coord)
    }

    pub fn remove(&mut self, chunk_coord: (i16, i16, i16)) -> (Entity, Handle<Mesh>) {
        #[cfg(feature = "debug")]
        {
            let result = self.0.remove(&chunk_coord);
            assert!(
                result.is_some(),
                "ChunkEntityMap::remove: chunk coord {chunk_coord:?} had no entity"
            );
            result.unwrap()
        }
        #[cfg(not(feature = "debug"))]
        {
            self.0.remove(&chunk_coord).unwrap()
        }
    }
}
