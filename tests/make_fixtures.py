"""Fabrique des fichiers témoins réalistes (iPhone P3 + GPS + rotation, HEIC, CMJN)."""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
ICC = Path(__file__).parent / "icc_dl"

def scene(w=600, h=400):
    a = np.zeros((h, w, 3), np.uint8)
    a[:] = (200, 190, 175)
    a[:, : w // 3] = (230, 20, 20)        # rouge saturé
    a[: h // 4, :] = (250, 250, 250)      # haut clair : repère d'orientation
    return a

def tag(path, *extra):
    subprocess.run(["exiftool", "-q", "-overwrite_original",
        "-GPSLatitude=43.6961", "-GPSLatitudeRef=N", "-GPSLongitude=7.2718", "-GPSLongitudeRef=E",
        "-Make=Apple", "-Model=iPhone 15 Pro", "-Artist=Proprietaire Test",
        "-XMP-dc:Description=12 rue de France Nice", "-IPTC:City=Nice", *extra, str(path)], check=True)

def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    p3 = (ICC / "DisplayP3-v4.icc").read_bytes()
    im = Image.fromarray(scene())
    im.save(out / "iphone_p3.jpg", quality=95, icc_profile=p3, comment=b"12 rue de France")
    tag(out / "iphone_p3.jpg", "-Orientation#=6")
    im.save(out / "iphone.heic", quality=90, icc_profile=p3)
    tag(out / "iphone.heic", "-Orientation#=6")
    Image.fromarray(scene()).convert("CMYK").save(out / "cmjn.jpg", quality=95)
    Image.fromarray(scene()).save(out / "srgb_sans_icc.jpg", quality=95)
    tag(out / "srgb_sans_icc.jpg")

if __name__ == "__main__":
    build(Path(sys.argv[1]))
