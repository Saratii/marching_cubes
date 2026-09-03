"""Generate assets/icons/copper.png: the 16x16 pixel-art nugget the HUD's
delivered-copper readout uses. Palette matches the ore tile's rock and vein."""

from PIL import Image

OUT = "assets/icons/copper.png"

PALETTE = {
    ".": (0, 0, 0, 0),
    "d": (58, 55, 52, 255),
    "r": (92, 87, 81, 255),
    "l": (126, 120, 111, 255),
    "o": (166, 79, 32, 255),
    "c": (226, 124, 52, 255),
    "b": (247, 178, 104, 255),
}

SPRITE = [
    "................",
    "........dddd....",
    "......ddllllrd..",
    ".....dlllcccld..",
    "....dlllcbbcld..",
    "...ddllcbbccrd..",
    "...dllccbccollld",
    "..ddlccoocolllld",
    "..dlrccoocllllld",
    "..dllcoocllllrld",
    "..dllcoolllllld.",
    "..dlloollllllrd.",
    "..ddlllllllllld.",
    "...dllllllllrd..",
    "....ddrlllldd...",
    "......dddddd....",
]


def main():
    assert len(SPRITE) == 16, "sprite must be 16 rows"
    image = Image.new("RGBA", (16, 16))
    for y, row in enumerate(SPRITE):
        assert len(row) == 16, f"row {y} is {len(row)} wide"
        for x, key in enumerate(row):
            image.putpixel((x, y), PALETTE[key])
    image.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
