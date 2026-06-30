"""
Shrink the raw Swedish-Leaf scans (large .tif) into a compact, Git-friendly set
of downsized .jpg files in place. Turns ~2 GB into ~10 MB with no loss that
matters (the model trains on 128x128 grayscale anyway).

Usage:
    python scripts/compress_dataset.py            # default: max side 700px, q85
    python scripts/compress_dataset.py 512 80     # custom max-side and JPEG quality

After running, if you want the compact dataset committed, remove the `data/`
line from .gitignore.
"""
import os
import sys
import glob
import cv2

DATA = "data"
MAX_SIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 700
QUALITY = int(sys.argv[2]) if len(sys.argv) > 2 else 85


def downsize(path):
    img = cv2.imread(path)
    if img is None:
        return 0, 0
    before = os.path.getsize(path)
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = os.path.splitext(path)[0] + ".jpg"
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if out != path:
        os.remove(path)
    return before, os.path.getsize(out)


def main():
    tifs = glob.glob(os.path.join(DATA, "*", "*.tif")) + glob.glob(os.path.join(DATA, "*", "*.tiff"))
    if not tifs:
        print("No .tif files found under data/. Nothing to do.")
        return
    total_before = total_after = 0
    for p in tifs:
        b, a = downsize(p)
        total_before += b
        total_after += a
    print("Compressed %d images: %.1f MB -> %.1f MB"
          % (len(tifs), total_before / 1e6, total_after / 1e6))


if __name__ == "__main__":
    main()
