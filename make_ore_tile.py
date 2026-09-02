"""Generate assets/source_tiles/ore.png: a seamless dark rock tile veined with
mineral. Run this before build_texture_ktx.py if you retune the constants."""

import numpy as np
from PIL import Image

SIZE = 400
SEED = 7
OUT = "assets/source_tiles/ore.png"

ROCK_DARK = np.array([46, 44, 43], dtype=np.float32)
ROCK_LIGHT = np.array([116, 111, 104], dtype=np.float32)
VEIN_DARK = np.array([104, 36, 14], dtype=np.float32)
VEIN_LIGHT = np.array([236, 128, 56], dtype=np.float32)

# oxidised copper, which is what stops the vein reading as gold
PATINA = np.array([50, 104, 84], dtype=np.float32)
PATINA_COVERAGE = 0.24
PATINA_STRENGTH = 0.55

# copper bleeding into the surrounding stone, so the metal is not a decal
STAIN = np.array([112, 62, 38], dtype=np.float32)
STAIN_COVERAGE = 0.45
STAIN_STRENGTH = 0.45

# fraction of the tile the mineral covers, and how sharp its edge is
VEIN_COVERAGE = 0.22
VEIN_EDGE = 0.018


def value_noise(rng, cells):
    """Tileable value noise: a wrapping lattice bilinearly interpolated with a
    smoothstep, so the left edge meets the right."""
    lattice = rng.random((cells, cells)).astype(np.float32)
    coords = np.arange(SIZE, dtype=np.float32) * cells / SIZE
    index = np.floor(coords).astype(np.int32)
    frac = coords - index
    frac = frac * frac * (3.0 - 2.0 * frac)
    i0 = index % cells
    i1 = (index + 1) % cells
    fy = frac[:, None]
    fx = frac[None, :]
    c00 = lattice[np.ix_(i0, i0)]
    c01 = lattice[np.ix_(i0, i1)]
    c10 = lattice[np.ix_(i1, i0)]
    c11 = lattice[np.ix_(i1, i1)]
    top = c00 + (c01 - c00) * fx
    bottom = c10 + (c11 - c10) * fx
    return top + (bottom - top) * fy


def fbm(rng, base_cells, octaves):
    total = np.zeros((SIZE, SIZE), dtype=np.float32)
    amplitude = 1.0
    weight = 0.0
    for octave in range(octaves):
        total += value_noise(rng, base_cells * 2**octave) * amplitude
        weight += amplitude
        amplitude *= 0.5
    return total / weight


def normalize(field):
    return (field - field.min()) / max(float(np.ptp(field)), 1e-6)


def main():
    rng = np.random.default_rng(SEED)
    rock = normalize(fbm(rng, 4, 5))
    grit = normalize(fbm(rng, 40, 2))
    rock = np.clip(rock * 0.75 + grit * 0.25, 0.0, 1.0)

    # veins cluster where a coarse mask and a fine mask agree, which reads as
    # mineral running through the stone rather than uniform speckle
    coarse = normalize(fbm(rng, 3, 2))
    fine = normalize(fbm(rng, 26, 4))
    vein_field = coarse * 0.45 + fine * 0.55
    threshold = np.quantile(vein_field, 1.0 - VEIN_COVERAGE)
    vein = np.clip((vein_field - threshold) / VEIN_EDGE, 0.0, 1.0)

    # the stain is the same field at a looser threshold, so it hugs the veins
    stain = np.clip(
        (vein_field - np.quantile(vein_field, 1.0 - STAIN_COVERAGE)) / 0.09, 0.0, 1.0
    ) * STAIN_STRENGTH

    sparkle = normalize(fbm(rng, 80, 3)) ** 1.6
    rock_color = ROCK_DARK + (ROCK_LIGHT - ROCK_DARK) * rock[..., None]
    rock_color = rock_color + (STAIN - rock_color) * stain[..., None]
    vein_color = VEIN_DARK + (VEIN_LIGHT - VEIN_DARK) * sparkle[..., None]

    # patina eats the duller parts of the vein, leaving bright metal exposed
    patina_field = normalize(fbm(rng, 12, 3)) * (1.0 - sparkle)
    patina = np.clip(
        (patina_field - np.quantile(patina_field, 1.0 - PATINA_COVERAGE)) / 0.12, 0.0, 1.0
    ) * PATINA_STRENGTH
    vein_color = vein_color + (PATINA - vein_color) * patina[..., None]

    image = rock_color + (vein_color - rock_color) * vein[..., None]

    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").save(OUT)
    print(f"wrote {OUT} ({SIZE}x{SIZE}), mineral covers {vein.mean():.1%}")


if __name__ == "__main__":
    main()
