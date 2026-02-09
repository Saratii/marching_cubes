from PIL import Image
import sys

GRID = 3

if len(sys.argv) != 2:
    print("usage: python preview_tile.py path/to/image.png")
    sys.exit(1)

path = sys.argv[1]

img = Image.open(path).convert("RGBA")
w, h = img.size

out = Image.new("RGBA", (w * GRID, h * GRID))

for y in range(GRID):
    for x in range(GRID):
        out.paste(img, (x * w, y * h))

out.show()
