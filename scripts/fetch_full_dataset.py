"""
Fetch the missing Swedish-Leaf classes from Linkoping University and add them to
data/ as compact downsized JPGs (the huge original .tif scans are discarded).

Processes one class at a time to keep peak disk usage low.
"""
import os
import sys
import glob
import zipfile
import shutil
import subprocess
import cv2

BASE = "https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/"
DATA = "data"
TMP = "_dl_tmp"
MAX_SIDE = 700
QUALITY = 85

# Only the classes not already present.
HAVE = {1, 2, 5, 7}
ALL = range(1, 16)
MISSING = [n for n in ALL if n not in HAVE]


def downsize(src, dst):
    img = cv2.imread(src)
    if img is None:
        return False
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return True


def fetch(n):
    dest_dir = os.path.join(DATA, "leaf%d" % n)
    if os.path.isdir(dest_dir) and len([f for f in os.listdir(dest_dir)]) >= 50:
        print("leaf%d already present, skipping." % n)
        return
    os.makedirs(TMP, exist_ok=True)
    zip_path = os.path.join(TMP, "leaf%d.zip" % n)
    url = BASE + "leaf%d.zip" % n
    print("Downloading leaf%d ..." % n, flush=True)
    r = subprocess.run(["curl", "-L", "-A", "Mozilla/5.0", "--retry", "3",
                        "--retry-delay", "5", "-C", "-", "-o", zip_path, url])
    if r.returncode != 0 or not os.path.exists(zip_path):
        print("  download FAILED for leaf%d" % n)
        return
    ex = os.path.join(TMP, "leaf%d_x" % n)
    os.makedirs(ex, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ex)
    except Exception as e:
        print("  unzip FAILED for leaf%d: %s" % (n, e))
        shutil.rmtree(TMP, ignore_errors=True)
        return
    imgs = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        imgs += glob.glob(os.path.join(ex, "**", ext), recursive=True)
    os.makedirs(dest_dir, exist_ok=True)
    kept = 0
    for i, src in enumerate(sorted(imgs)):
        if downsize(src, os.path.join(dest_dir, "l%dnr%03d.jpg" % (n, i + 1))):
            kept += 1
    # cleanup this class's temp data immediately
    os.remove(zip_path)
    shutil.rmtree(ex, ignore_errors=True)
    print("  leaf%d: %d images saved to %s" % (n, kept, dest_dir), flush=True)


def main():
    for n in MISSING:
        fetch(n)
    shutil.rmtree(TMP, ignore_errors=True)
    print("Done. Classes now present:",
          sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))))


if __name__ == "__main__":
    main()
