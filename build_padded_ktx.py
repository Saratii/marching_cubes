from PIL import Image
import os
import subprocess

TILES = [
    "assets/source_tiles/dirt.png",
    "assets/source_tiles/grass3.png",
    "assets/source_tiles/sand.png",
]

TOKTX = r"C:\Program Files\KTX-Software\bin\toktx.exe"
OUT_ARRAY = "assets/texture_array.ktx2"

# Save all tiles as separate files (they must be same dimensions)
temp_tiles = []
for i, path in enumerate(TILES):
    img = Image.open(path).convert("RGBA")
    temp_path = f"temp_tile_{i}.png"
    img.save(temp_path)
    temp_tiles.append(temp_path)

# Create texture array with mipmaps
cmd = [
    TOKTX,
    "--genmipmap",
    "--layers", str(len(TILES)),
    "--t2",
    OUT_ARRAY,
] + temp_tiles

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# Cleanup
for temp in temp_tiles:
    os.remove(temp)

print(f"Created texture array: {OUT_ARRAY}")