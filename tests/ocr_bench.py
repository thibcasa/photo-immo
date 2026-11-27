"""Banc OCR : rappel de détection d'adresses/plaques sur scènes synthétiques variées (taille, angle,
contraste, texte clair sur fond sombre). Usage : python3 tests/ocr_bench.py  → compare ancien/nouveau."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
TEXTES = ["12 rue de France", "47 avenue Jean Médecin", "AB-123-CD", "06000 Nice", "3 bd Gambetta",
          "Résidence Les Mimosas", "FG-456-HJ", "8 place Garibaldi"]


def scene(i: int, rng) -> tuple[np.ndarray, str]:
    W, H = 1600, 1200
    base = rng.uniform(0.55, 0.85)
    bg = np.clip(base + rng.normal(0, 0.03, (H // 8, W // 8, 1)), 0, 1)
    bg = np.asarray(Image.fromarray((np.repeat(bg, 3, 2) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC))
    im = Image.fromarray(bg)
    txt = TEXTES[i % len(TEXTES)]
    size = [18, 24, 32, 44][i % 4]
    angle = [0, 6, -9, 12, -4, 0][i % 6]
    dark_plate = i % 5 == 0          # texte clair sur fond sombre (plaque de rue émaillée)
    low_contrast = i % 7 == 3
    f = ImageFont.truetype(FONTS[i % len(FONTS)], size)
    pad = size // 2
    w = int(f.getlength(txt)) + 2 * pad
    h = size + 2 * pad
    tile = Image.new("RGB", (w, h), (30, 50, 110) if dark_plate else (235, 235, 230))
    fill = (240, 240, 240) if dark_plate else ((150, 150, 150) if low_contrast else (20, 20, 20))
    ImageDraw.Draw(tile).text((pad, pad // 2), txt, font=f, fill=fill)
    tile = tile.rotate(angle, expand=True, fillcolor=tuple(int(v) for v in bg[0, 0]))
    x, y = int(rng.uniform(50, W - tile.width - 50)), int(rng.uniform(50, H - tile.height - 50))
    im.paste(tile, (x, y))
    return np.asarray(im).astype(np.float32) / 255, txt


def ancien(img):
    """OCR de la v2.1 (référence figée) : une passe Tesseract psm 11 sur l'image entière."""
    import subprocess, tempfile, cv2
    from rv2.ocr import _ADDRESS, _PLATE
    g = (img @ np.array([0.2126, 0.7152, 0.0722]) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        cv2.imwrite(td + "/o.png", g)
        out = subprocess.run(["tesseract", td + "/o.png", "stdout", "-l", "fra+eng", "--psm", "11"],
                             capture_output=True, text=True).stdout
    t = " ".join(w for w in out.split() if len(w) > 1)
    return {"adresse_suspectee": bool(_ADDRESS.search(t)), "plaque_suspectee": bool(_PLATE.search(t.upper()))}


def run(n: int = 24):
    from rv2.ocr import scan as nouveau
    rng = np.random.default_rng(7)
    res = {"ancien": 0, "nouveau": 0}
    for i in range(n):
        img, txt = scene(i, rng)
        a = ancien(img)
        b = nouveau(img)
        res["ancien"] += bool(a.get("adresse_suspectee") or a.get("plaque_suspectee"))
        res["nouveau"] += bool(b.get("adresse_suspectee") or b.get("plaque_suspectee"))
    return {k: f"{v}/{n} ({v / n:.0%})" for k, v in res.items()}


if __name__ == "__main__":
    print(run())
