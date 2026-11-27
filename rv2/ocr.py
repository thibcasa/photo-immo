"""OCR renforcé (§5 adresses/plaques, §10 textes injonctifs).

Passes Tesseract (fra+eng), mots retenus si confiance ≥ 45 :
  1. image entière (≤ 3000 px), mode texte épars ;
  2. zones de texte candidates (gradient morphologique + fermeture horizontale) recadrées, agrandies ×3,
     contraste égalisé (CLAHE), lues à 0° et ±10°, et en négatif (texte clair sur fond sombre) ;
Le rappel est MESURÉ sur un banc synthétique (tests/test_ocr_banc.py) : il est élevé mais pas total —
un texte minuscule, très incliné ou flou peut échapper au contrôle."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .io import to_uint8

CONF_MIN = 45
_ADDRESS = re.compile(
    r"\b(rue|avenue|av|bd|boulevard|chemin|all[ée]e|place|quai|impasse|route|promenade|cours|square|"
    r"traverse|mont[ée]e|villa|r[ée]sidence|b[aâ]t(iment)?|esc(alier)?|appt|appartement)\b|\b0[0-9]\d{3}\b|\b\d{5}\b", re.I)
_PLATE = re.compile(r"\b[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}\b|\b\d{1,4}\s?[A-Z]{2,3}\s?\d{2}\b")
_IMPERATIVE = re.compile(r"\b(ignore[rz]?|oublie[rz]?|supprime[rz]?|efface[rz]?|publie[rz]?|envoie[rz]?|"
                         r"instructions?|ex[ée]cute[rz]?|prompt|system|disregard|delete|send)\b", re.I)


def _tess(gray8: np.ndarray, psm: int) -> list[tuple[str, float, tuple]]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.png"
        cv2.imwrite(str(p), gray8)
        r = subprocess.run(["tesseract", str(p), "stdout", "-l", "fra+eng", "--psm", str(psm), "tsv"],
                           capture_output=True, text=True)
    words = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[11].strip():
            try:
                conf = float(f[10])
            except ValueError:
                continue
            if conf >= CONF_MIN:
                words.append((f[11].strip(), conf, (int(f[6]), int(f[7]), int(f[8]), int(f[9]))))
    return words


def _text_regions(gray8: np.ndarray, max_regions: int = 15) -> list[tuple[int, int, int, int]]:
    g = cv2.GaussianBlur(gray8, (3, 3), 0)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    otsu, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thr = max(otsu, float(np.percentile(grad, 97)))   # robuste à un fond texturé (marbre, crépi)
    bw = (grad > thr).astype(np.uint8) * 255
    H, W = g.shape
    kx = max(9, W // 120)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(closed)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h < 6 or h > H * 0.12 or w < 2 * h or w > W * 0.95:
            continue
        fill = area / (w * h)
        if fill < 0.35:
            continue
        cand.append((w * h, (x, y, w, h)))
    cand.sort(reverse=True)
    return [c[1] for c in cand[:max_regions]]


def _rotate(g: np.ndarray, deg: float) -> np.ndarray:
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(g, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def _skew(crop: np.ndarray) -> float:
    """Angle du texte dans la zone [mesuré] : rectangle d'aire minimale des pixels de texte."""
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw
    pts = cv2.findNonZero(bw)
    if pts is None or len(pts) < 20:
        return 0.0
    (_, _), (w, h), a = cv2.minAreaRect(pts)
    if w < h:
        a = a - 90
    a = ((a + 45) % 90) - 45
    return float(a)


def read_text(img: np.ndarray, angles=(0, -6, 6)) -> list[tuple[str, float]]:
    g = to_uint8(img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) if img.ndim == 3 else img
    H, W = g.shape
    s = min(1.0, 3000 / max(H, W))
    g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else g
    words = [(w, c) for w, c, _ in _tess(g, 11)]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    for x, y, w, h in _text_regions(g):
        pad = int(0.6 * h)
        crop = g[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crop = clahe.apply(crop)
        sk = _skew(crop)
        for a in sorted({round(sk, 1), *angles}, key=lambda v: abs(v - sk)):
            r = _rotate(crop, a) if a else crop
            for variant in (r, 255 - r):
                words += [(t, c) for t, c, _ in _tess(variant, 6)]
    return words


def scan(img: np.ndarray) -> dict:
    if shutil.which("tesseract") is None:
        return {"texte_lu": "", "adresse_suspectee": None, "plaque_suspectee": None, "texte_injonctif": None,
                "source": "tesseract absent : contrôle humain requis"}
    words = read_text(img)
    seen, uniq = set(), []
    for w, c in words:
        k = w.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    text = " ".join(uniq)
    return {"texte_lu": text[:800],
            "adresse_suspectee": bool(_ADDRESS.search(text)),
            "plaque_suspectee": bool(_PLATE.search(text.upper())),
            "texte_injonctif": bool(_IMPERATIVE.search(text)),
            "source": "mesuré — OCR Tesseract multi-passes (rappel mesuré sur banc, non exhaustif)"}
