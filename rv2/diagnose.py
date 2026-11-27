"""DIAGNOSTIC (§6.1). Chaque valeur porte son étiquette de source."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .io import to_uint8

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def M(value, source: str, unit: str = "") -> dict:
    """Valeur étiquetée : source ∈ {'mesuré', 'estimé', 'mesuré — indicatif'}."""
    return {"valeur": value, "source": source, "unite": unit}


# ----------------------------------------------------------------- histogramme
def histogram(img: np.ndarray) -> dict:
    y8 = to_uint8(luminance(img))
    rgb8 = to_uint8(img)
    return {
        "brules_pct": M(round(float((y8 >= 250).mean() * 100), 2), "mesuré", "% px luminance ≥ 250/255"),
        "bouches_pct": M(round(float((y8 <= 5).mean() * 100), 2), "mesuré", "% px luminance ≤ 5/255"),
        "canal_sature_pct": M(round(float((rgb8 >= 255).any(axis=2).mean() * 100), 2), "mesuré",
                              "% px avec au moins un canal à 255"),
        "luminance_mediane": M(round(float(np.median(y8)), 1), "mesuré", "/255"),
    }


# ----------------------------------------------------------------- dominante
def _neutral_mask(img: np.ndarray) -> np.ndarray:
    y = luminance(img)
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    m = (y > 0.12) & (y < 0.92) & (sat < 0.35)
    return m if m.sum() > 500 else (y > 0.05) & (y < 0.97)


def estimate_cct(img: np.ndarray) -> dict:
    """Température de couleur MOYENNE de la scène (McCamy) : toujours [estimé] — ce n'est pas
    une mesure de l'éclairage (seule une mire de gris le permettrait)."""
    lin = srgb_to_linear(img)[_neutral_mask(img)]
    r, g, b = lin.mean(axis=0)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    s = X + Y + Z
    x, yv = X / s, Y / s
    n = (x - 0.3320) / (0.1858 - yv)
    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    cct = int(round(float(np.clip(cct, 1500, 15000)) / 50) * 50)
    return {"temperature_K": M(cct, "estimé", "K (moyenne scène, méthode McCamy)"),
            "ratio_R_G": M(round(float(r / g), 3), "mesuré", "moyenne zones neutres"),
            "ratio_B_G": M(round(float(b / g), 3), "mesuré", "moyenne zones neutres")}


def patch_neutrality(img: np.ndarray, rect: tuple[int, int, int, int]) -> dict:
    """Mire de gris / zone neutre désignée : écart mesuré des canaux."""
    x, y, w, h = rect
    p = srgb_to_linear(img[y:y + h, x:x + w]).reshape(-1, 3).mean(axis=0)
    return {"ecart_max_canaux_pct": M(round(float((p.max() - p.min()) / p.mean() * 100), 2), "mesuré", "%")}


# ----------------------------------------------------------------- verticales
def detect_vertical_lines(img: np.ndarray, tol_deg: float = 20.0, max_dim: int = 1600):
    g = to_uint8(luminance(img))
    H, W = g.shape
    s = min(1.0, max_dim / max(H, W))
    if s < 1:
        g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    h = g.shape[0]
    edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=50,
                            minLineLength=max(20, int(0.08 * h)), maxLineGap=max(3, int(0.01 * h)))
    out = []
    if lines is None:
        return out
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4 : (N,1,4) ; OpenCV 5 : (N,4)
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dy <= 0:
            continue
        ang = float(np.degrees(np.arctan2(dx, dy)))  # >0 : la ligne part vers la droite en descendant
        if abs(ang) <= tol_deg:
            out.append({"x_mid": (x1 + x2) / 2 / s, "y_mid": (y1 + y2) / 2 / s, "angle": ang,
                        "longueur": float(np.hypot(dx, dy)) / s})
    return out


def measure_verticals(img: np.ndarray) -> dict:
    """Modèle angle(x) = a·(x − W/2) + b, pondéré par la longueur, une passe de rejet des aberrants.
    b = inclinaison globale (roulis) ; |a|·W/2 = convergence en bord d'image (perspective)."""
    W = img.shape[1]
    lines = detect_vertical_lines(img)
    if len(lines) < 4:
        return {"n_lignes": len(lines), "fiable": False,
                "inclinaison_deg": M(None, "mesuré", "non mesurable (< 4 lignes)"),
                "convergence_bords_deg": M(None, "mesuré", "non mesurable")}
    x = np.array([l["x_mid"] for l in lines]) - W / 2
    a = np.array([l["angle"] for l in lines])
    w = np.array([l["longueur"] for l in lines])
    keep = np.ones(len(a), bool)
    for _ in range(2):
        A = np.vstack([x[keep], np.ones(keep.sum())]).T
        sw = np.sqrt(w[keep])
        coef, *_ = np.linalg.lstsq(A * sw[:, None], a[keep] * sw, rcond=None)
        res = np.abs(a - (coef[0] * x + coef[1]))
        new_keep = res < max(1.0, 2.5 * np.median(res[keep]) + 0.2)
        if new_keep.sum() < 4 or (new_keep == keep).all():
            break
        keep = new_keep
    slope, tilt = coef
    conv = float(slope * W / 2)
    res_k = np.abs(a - (slope * x + tilt))[keep]
    dispersion = float(np.average(res_k, weights=w[keep]))
    return {"n_lignes": int(keep.sum()),
            "dispersion_deg": M(round(dispersion, 2), "mesuré", "° (écart moyen des lignes au modèle)"),
            "fiable": bool(keep.sum() >= 6 and dispersion <= 0.6 and keep.sum() >= 0.5 * len(a)),
            "inclinaison_deg": M(round(float(tilt), 2), "mesuré", "° (Hough, centre image)"),
            "convergence_bords_deg": M(round(conv, 2), "mesuré", "° (écart des verticales en bord d'image)"),
            "_slope": float(slope), "_tilt": float(tilt)}


# ----------------------------------------------------------------- netteté / bruit
def sharpness(img: np.ndarray, grid: int = 4) -> dict:
    g = to_uint8(luminance(img)).astype(np.float32)
    H, W = g.shape
    lap = cv2.Laplacian(g, cv2.CV_32F)
    tiles = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            tiles[i, j] = lap[i * H // grid:(i + 1) * H // grid, j * W // grid:(j + 1) * W // grid].var()
    ref = np.percentile(tiles, 75)
    molles = [f"L{i + 1}C{j + 1}" for i in range(grid) for j in range(grid) if tiles[i, j] < 0.08 * ref]
    return {"variance_laplacien": M(round(float(lap.var()), 1), "mesuré", "global"),
            "zones_molles": M(molles, "mesuré — indicatif",
                              "tuiles 4×4 ; un mur lisse ou un ciel ressort aussi : vérifier à 100 %")}


def noise_sigma(img: np.ndarray) -> dict:
    """Écart-type du bruit, méthode d'Immerkær (1996), échelle 0–255."""
    g = to_uint8(luminance(img)).astype(np.float64)
    H, W = g.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    sigma = np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)) * np.abs(conv).sum()
    return {"bruit_sigma": M(round(float(sigma), 2), "mesuré", "/255 (Immerkær)")}


# ----------------------------------------------------------------- texte visible (OCR)
_ADDRESS = re.compile(
    r"(\b\d{1,4}\s*(bis|ter)?\s*,?\s*)?\b(rue|avenue|av\.?|bd|boulevard|chemin|all[ée]e|place|quai|impasse|"
    r"route|promenade|cours|square|traverse|mont[ée]e|villa|r[ée]sidence)\b|\b0[0-9]\d{3}\b|\b\d{5}\b", re.I)
_PLATE = re.compile(r"\b[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}\b")
_IMPERATIVE = re.compile(r"\b(ignore[rz]?|oublie[rz]?|supprime[rz]?|efface[rz]?|publie[rz]?|envoie[rz]?|"
                         r"instructions?|ex[ée]cute[rz]?|prompt|system)\b", re.I)


def ocr_scan(img: np.ndarray) -> dict:
    """Texte visible (§5 adresse/plaques, §10 sécurité des entrées). Le texte lu est une DONNÉE :
    il n'est jamais exécuté, seulement signalé."""
    if shutil.which("tesseract") is None:
        return {"ocr": M(None, "mesuré", "tesseract absent : contrôle humain requis")}
    g = to_uint8(luminance(img))
    if max(g.shape) > 3000:
        s = 3000 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ocr.png"
        cv2.imwrite(str(p), g)
        out = subprocess.run(["tesseract", str(p), "stdout", "-l", "fra+eng", "--psm", "11"],
                             capture_output=True, text=True)
    words = [w for w in out.stdout.split() if len(w) > 1]
    text = " ".join(words)
    return {
        "texte_lu": text[:500],
        "adresse_suspectee": bool(_ADDRESS.search(text)),
        "plaque_suspectee": bool(_PLATE.search(text.upper())),
        "texte_injonctif": bool(_IMPERATIVE.search(text)),
        "source": "mesuré — OCR Tesseract, non exhaustif (texte incliné/petit possiblement manqué)",
    }


# ----------------------------------------------------------------- synthèse
def diagnose(img: np.ndarray, info: dict | None = None, meta: dict | None = None, ocr: bool = True) -> dict:
    info = info or {}
    d = {"dimensions": M(f"{img.shape[1]}×{img.shape[0]}", "mesuré", "px")}
    d.update(histogram(img))
    d.update(estimate_cct(img))
    v = measure_verticals(img)
    d.update({k: val for k, val in v.items() if not k.startswith("_")})
    d["verticales_fiables"] = M(v["fiable"], "mesuré", "≥ 6 lignes concordantes")
    d.update(sharpness(img))
    d.update(noise_sigma(img))
    if info.get("jpeg_qualite_estimee") is not None:
        d["jpeg_qualite_source"] = M(info["jpeg_qualite_estimee"], "estimé", "équiv. IJG (table de quantif.)")
    if meta:
        f35 = meta.get("ExifIFD:FocalLengthIn35mmFormat")
        d["focale_35mm"] = M(f35, "mesuré", "mm (EXIF)") if f35 else M(None, "mesuré", "absente de l'EXIF")
        d["gps_dans_source"] = M(any("GPS" in k for k in meta), "mesuré", "EXIF")
    if ocr:
        d["texte_visible"] = ocr_scan(img)
    d["personnes_ecrans_documents"] = M(None, "estimé",
                                        "aucun détecteur automatique fiable ici : contrôle visuel obligatoire")
    return d
