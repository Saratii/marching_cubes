from PIL import Image
import os
import subprocess

MAX_MIPS = "3"
TILES = [
    "assets/source_tiles/dirt.png",
    "assets/source_tiles/grass.png",
]

PAD = 2
OUT_PADDED_DIR = "assets/padded_tiles"
ATLAS_PNG = "assets/texture_atlas_padded.png"
ATLAS_KTX2 = "assets/texture_atlas.ktx2"

os.makedirs(OUT_PADDED_DIR, exist_ok=True)

padded_tiles = []

for i, path in enumerate(TILES):
    print("Loading:", path)
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    pw, ph = w + PAD*2, h + PAD*2

    padded = Image.new("RGBA", (pw, ph))
    padded.paste(img, (PAD, PAD))

    # edges
    left   = img.crop((0,     0,     1,     h)).resize((PAD, h))
    right  = img.crop((w - 1, 0,     w,     h)).resize((PAD, h))
    top    = img.crop((0,     0,     w,     1)).resize((w, PAD))
    bottom = img.crop((0,     h - 1, w,     h)).resize((w, PAD))

    padded.paste(left,   (0,      PAD))
    padded.paste(right,  (PAD+w,  PAD))
    padded.paste(top,    (PAD,    0))
    padded.paste(bottom, (PAD,    PAD+h))

    # corners
    tl = img.crop((0,     0,     1,     1)).resize((PAD, PAD))
    tr = img.crop((w - 1, 0,     w,     1)).resize((PAD, PAD))
    bl = img.crop((0,     h - 1, 1,     h)).resize((PAD, PAD))
    br = img.crop((w - 1, h - 1, w,     h)).resize((PAD, PAD))

    padded.paste(tl, (0,        0))
    padded.paste(tr, (PAD + w,  0))
    padded.paste(bl, (0,        PAD + h))
    padded.paste(br, (PAD + w,  PAD + h))

    out_tile = os.path.join(OUT_PADDED_DIR, f"tile_{i}.png")
    padded.save(out_tile)
    padded_tiles.append(padded)

print("Created padded tiles.")

# assemble atlas horizontally
pw, ph = padded_tiles[0].size
atlas = Image.new("RGBA", (pw * len(padded_tiles), ph))

x = 0
for tile in padded_tiles:
    atlas.paste(tile, (x, 0))
    x += pw

atlas.save(ATLAS_PNG)
print("Atlas saved:", ATLAS_PNG)

# KTX2 with mipmaps
TOKTX = r"C:\Program Files\KTX-Software\bin\toktx.exe"

cmd = [
    TOKTX,
    "--genmipmap",
    "--t2",
    ATLAS_KTX2,
    ATLAS_PNG
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Saved:", ATLAS_KTX2)

#& "C:\Program Files\KTX-Software\bin\ktx.exe" extract --level all assets\texture_atlas.ktx2 assets\mipdump