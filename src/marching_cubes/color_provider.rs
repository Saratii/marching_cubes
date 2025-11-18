use bevy::prelude::Vec3;
pub trait ColorProvider: Send + Sync {
    fn get_color(&self, material: u8, position: Vec3) -> [f32; 4];

    fn needs_material(&self) -> bool {
        false
    }

    fn uses_normals(&self) -> bool {
        false
    }
}

pub struct NormalColorProvider;

impl ColorProvider for NormalColorProvider {
    fn get_color(&self, _material: u8, _position: Vec3) -> [f32; 4] {
        [1.0, 1.0, 1.0, 1.0]
    }

    fn needs_material(&self) -> bool {
        false
    }

    fn uses_normals(&self) -> bool {
        true
    }
}

pub fn normal_to_color(normal: Vec3) -> [f32; 4] {
    let nc = normal * 0.5 + Vec3::ONE * 0.5;
    [nc.x, nc.y, nc.z, 1.0]
}

pub struct MaterialColorProvider;

impl ColorProvider for MaterialColorProvider {
    fn get_color(&self, material: u8, _position: Vec3) -> [f32; 4] {
        match material {
            1 => [73. / 255., 34. / 255., 1. / 255., 1.0],
            2 => [17. / 255., 124. / 255., 19. / 255., 1.0],
            255 => [1.0, 0.0, 1.0, 1.0],
            _ => [0.5, 0.5, 0.5, 1.0],
        }
    }

    fn needs_material(&self) -> bool {
        true
    }
}

pub struct PositionColorProvider {
    pub scale: f32,
}

impl Default for PositionColorProvider {
    fn default() -> Self {
        Self { scale: 0.01 }
    }
}

impl ColorProvider for PositionColorProvider {
    fn get_color(&self, _material: u8, position: Vec3) -> [f32; 4] {
        let r = ((position.x * self.scale).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g = ((position.y * self.scale).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b = ((position.z * self.scale).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        [r, g, b, 1.0]
    }

    fn needs_material(&self) -> bool {
        false
    }
}

pub struct ConstantColorProvider {
    pub color: [f32; 4],
}

impl ConstantColorProvider {
    pub fn new(color: [f32; 4]) -> Self {
        Self { color }
    }
}

impl ColorProvider for ConstantColorProvider {
    fn get_color(&self, _material: u8, _position: Vec3) -> [f32; 4] {
        self.color
    }

    fn needs_material(&self) -> bool {
        false
    }
}
