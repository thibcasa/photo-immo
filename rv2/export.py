"""EXPORT (§8) : formats, mention ORANGE incrustée APRÈS le recadrage de chaque format
(donc jamais rognée), écriture JPEG sRGB + ICC sans EXIF/GPS/XMP, contrôle [mesuré] après écriture."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import from_uint8, save_jpeg, to_uint8, verify_delivered_file

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
MENTION_MIN_RATIO = 0.03     # §5 : hauteur du texte ≥ 3 % de la hauteur de l'image (seuil interne)
MENTION_CIBLE_RATIO = 0.034  # marge de sécurité de l'outil
CONTRASTE_MIN = 4.5          # référence WCAG AA (lisibilité) — référence technique, pas une obligation légale
MIN_LARGEUR_PORTAIL = 2400   # §8, cible interne

FORMATS = {
    "principal": None,                      # VENTE : 3:2 ou 4:3 paysage / LCD : paysage
    "ig45": (1080, 1350),
    "ig11": (1080, 1080),
    "story": (1080, 1920),
}
PERTE_RECADRAGE_MAX = 0.25   # au-delà, on ne rogne pas : bandes unies (aucune partie de la pièce masquée)
FOND_BANDES = (0.10, 0.10, 0.10)


def _font_path() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("aucune police TrueType disponible pour la mention")


def _rel_lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722])


def apply_mention(img: np.ndarray, text: str) -> tuple[np.ndarray, dict]:
    """Bandeau bas, texte blanc sur fond sombre semi-opaque. Mesure ce qui a été réellement dessiné :
    hauteur du texte (px et % de l'image), position dans le cadre, contraste."""
    H, W = img.shape[:2]
    im = Image.fromarray(to_uint8(img), "RGB").convert("RGBA")
    fp = _font_path()
    target_h = max(MENTION_MIN_RATIO, MENTION_CIBLE_RATIO) * H
    size = max(8, int(target_h))
    lines = [text]

    def measure(sz, lns):
        f = ImageFont.truetype(fp, sz)
        boxes = [f.getbbox(l) for l in lns]
        return f, boxes

    max_w = 0.92 * W

    def wrap(f):
        words, out_lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            bb = f.getbbox(trial)
            if bb[2] - bb[0] <= max_w or not cur:
                cur = trial
            else:
                out_lines.append(cur)
                cur = w
        out_lines.append(cur)
        return out_lines

    # taille croissante jusqu'à ce que CHAQUE ligne (après coupure éventuelle) atteigne la hauteur cible ;
    # on ne réduit jamais sous la cible pour faire tenir le texte
    for _ in range(200):
        f = ImageFont.truetype(fp, size)
        lines = wrap(f)
        f, boxes = measure(size, lines)
        if min(b[3] - b[1] for b in boxes) >= target_h:
            break
        size += 1
    if len(lines) > 3 or max(b[2] - b[0] for b in boxes) > max_w:
        raise ValueError("mention impossible à rendre lisible dans ce format : format non livrable")

    pad = int(0.35 * size)
    line_gap = int(0.25 * size)
    heights = [b[3] - b[1] for b in boxes]
    block_h = sum(heights) + line_gap * (len(lines) - 1)
    margin_bottom = int(0.025 * H)
    y_top = H - margin_bottom - block_h - 2 * pad
    widths = [b[2] - b[0] for b in boxes]
    band_w = max(widths) + 2 * pad
    x_band = (W - band_w) // 2

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([x_band, y_top, x_band + band_w, y_top + block_h + 2 * pad], fill=(0, 0, 0, 175))
    y = y_top + pad
    drawn = []
    for line, b, hgt in zip(lines, boxes, heights):
        x = (W - (b[2] - b[0])) // 2 - b[0]
        d.text((x, y - b[1]), line, font=f, fill=(255, 255, 255, 255))
        drawn.append((x + b[0], y, x + b[2], y + hgt))
        y += hgt + line_gap
    out = Image.alpha_composite(im, overlay).convert("RGB")
    arr = np.asarray(out)

    # mesures sur les pixels rendus
    band = arr[y_top:y_top + block_h + 2 * pad, x_band:x_band + band_w].reshape(-1, 3)
    lum = _rel_lum(band)
    text_l = np.percentile(lum, 99.5)
    bg_l = np.percentile(lum, 60)
    contrast = (text_l + 0.05) / (bg_l + 0.05)
    inside = all(0 <= x0 and 0 <= y0 and x1 <= W and y1 <= H for x0, y0, x1, y1 in drawn)
    h_pct = min(heights) / H * 100
    rep = {"texte": text, "lignes": len(lines), "hauteur_texte_px": min(heights),
           "hauteur_texte_pct": round(h_pct, 2), "dans_le_cadre": inside, "contraste": round(float(contrast), 2),
           "conforme": bool(h_pct >= MENTION_MIN_RATIO * 100 and inside and contrast >= CONTRASTE_MIN),
           "source": "mesuré (rendu effectif)"}
    return from_uint8(arr), rep


def _fit(img: np.ndarray, size: tuple[int, int], center_x: float = 0.5) -> tuple[np.ndarray, dict]:
    H, W = img.shape[:2]
    tw, th = size
    tr, r = tw / th, W / H
    if r > tr:   # trop large : on rognerait les côtés
        new_w = int(round(H * tr))
        perte = 1 - new_w / W
        if perte <= PERTE_RECADRAGE_MAX:
            x0 = int(np.clip(center_x * W - new_w / 2, 0, W - new_w))
            crop = img[:, x0:x0 + new_w]
            return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA), {"mode": "recadrage", "perte_pct": round(perte * 100, 1)}
        scale = tw / W
        nh = int(round(H * scale))
        canvas = np.empty((th, tw, 3), np.float32)
        canvas[:] = FOND_BANDES
        y0 = (th - nh) // 2
        canvas[y0:y0 + nh] = cv2.resize(img, (tw, nh), interpolation=cv2.INTER_AREA)
        return canvas, {"mode": "bandes (image entière)", "perte_pct": 0.0}
    new_h = int(round(W / tr))
    perte = 1 - new_h / H
    if perte <= PERTE_RECADRAGE_MAX:
        y0 = (H - new_h) // 2
        return cv2.resize(img[y0:y0 + new_h], (tw, th), interpolation=cv2.INTER_AREA), {"mode": "recadrage", "perte_pct": round(perte * 100, 1)}
    scale = th / H
    nw = int(round(W * scale))
    canvas = np.empty((th, tw, 3), np.float32)
    canvas[:] = FOND_BANDES
    x0 = (tw - nw) // 2
    canvas[:, x0:x0 + nw] = cv2.resize(img, (nw, th), interpolation=cv2.INTER_AREA)
    return canvas, {"mode": "bandes (image entière)", "perte_pct": 0.0}


def principal(img: np.ndarray, usage: str, center_x: float = 0.5) -> tuple[np.ndarray, dict]:
    """VENTE : 3:2 ou 4:3 paysage (ratio natif conservé s'il est à ±1 %, sinon recadrage vers le plus
    proche). LCD : paysage, ratio natif. Jamais d'agrandissement (§5 : pas d'agrandissement inventé)."""
    H, W = img.shape[:2]
    rep = {"alertes": []}
    if W < H:
        rep["alertes"].append("photo verticale : format portail/LCD paysage non respecté (§8)")
        return img, rep | {"mode": "natif"}
    if usage == "VENTE":
        r = W / H
        target = min((3 / 2, 4 / 3), key=lambda t: abs(r - t))
        if abs(r - target) / target > 0.01:
            img, fit = _fit(img, (int(round(H * target)), H) if r > target else (W, int(round(W / target))), center_x)
            rep.update(fit)
            if fit["mode"].startswith("bandes"):
                rep["alertes"].append("ratio hors 3:2/4:3 et recadrage > 25 % : bandes ajoutées, à valider")
        else:
            rep["mode"] = "natif"
        rep["ratio"] = "3:2" if target == 1.5 else "4:3"
    else:
        rep["mode"] = "natif"
    if img.shape[1] < MIN_LARGEUR_PORTAIL:
        rep["alertes"].append(f"largeur {img.shape[1]} px < {MIN_LARGEUR_PORTAIL} px (§8) — pas d'agrandissement")
    return img, rep


def export_all(img: np.ndarray, out_dir: Path, name: str, usage: str, formats: list[str],
               mention: str | None, center_x: float = 0.5) -> list[dict]:
    """Produit chaque format, appose la mention APRÈS mise au format, écrit, puis vérifie le fichier."""
    results = []
    for fmt in formats:
        if fmt == "principal":
            canvas, fit = principal(img, usage, center_x)
            path = out_dir / name
        else:
            canvas, fit = _fit(img, FORMATS[fmt], center_x)
            path = out_dir / fmt / name
        res = {"format": fmt, "fichier": str(path), "mise_au_format": fit, "livrable": True, "alertes": list(fit.get("alertes", []))}
        if mention:
            try:
                canvas, mrep = apply_mention(canvas, mention)
            except ValueError as exc:
                res.update(livrable=False, motif=str(exc))
                results.append(res)
                continue
            res["mention"] = mrep
            if not mrep["conforme"]:
                res.update(livrable=False, motif="mention non conforme (taille, cadre ou contraste)")
                results.append(res)
                continue
        save_jpeg(canvas, path, quality=90)
        res["controle_fichier"] = verify_delivered_file(path)
        if not (res["controle_fichier"]["metadonnees_ok"] and res["controle_fichier"]["icc_srgb"]):
            res.update(livrable=False, motif="métadonnées résiduelles ou ICC sRGB absent")
            path.unlink(missing_ok=True)
        results.append(res)
    return results
