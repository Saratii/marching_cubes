from PIL import Image
import sys

if len(sys.argv) != 4:
    print("usage: python resize.py assets/raw/gravelly_sand_diff_4k.jpg assets/source_tiles/sand2.png 400 ")
    sys.exit(1)

inp, outp, size_str = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    size = int(size_str)
except ValueError:
    raise RuntimeError("SIZE must be an integer")

img = Image.open(inp)
w, h = img.size

if w < size or h < size:
    raise RuntimeError(f"image too small: {w}x{h}, need at least {size}x{size}")

img = img.resize((size, size), Image.LANCZOS)
img.save(outp)

print("saved:", outp)
