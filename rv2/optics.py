"""OPTIQUE / DISTORSION (§4, §6.3 : première étape).

1. Profil Lensfun (base libre, CC BY-SA) si boîtier ET objectif y figurent : correction exacte.
2. Sinon, estimation « fil à plomb » : on cherche le coefficient radial k (modèle x_d = x_u·(1 + k·r²),
   r normalisé par la demi-diagonale) qui MAXIMISE la longueur totale de lignes droites détectées
   (Hough) dans l'image corrigée. Appliqué seulement si le gain est net (≥ 10 %) et si la perte de cadre
   reste ≤ 15 % (sinon la correction masquerait une partie de la pièce, §5).
3. Focale EXIF équivalente 24×36 comparée à la cible §4 (16–24 mm) : en dessous de 14 mm, l'ultra
   grand-angle étire les bords → À REPRENDRE (constaté : Insta360 X5 recadré, 6 mm équiv.).
"""
from __future__ import annotations

import cv2
import numpy as np

from .io import to_uint8

GAIN_MIN = 0.10
PERTE_MAX_PCT = 15.0
FOCALE_MIN_35MM = 14.0  # seuil technique de l'outil (§4 vise 16–24 mm)


# ----------------------------------------------------------------- Lensfun
def lensfun_correct(img: np.ndarray, meta: dict | None):
    if not meta:
        return None, {"lensfun": "métadonnées absentes"}
    try:
        import lensfunpy
    except ImportError:
        return None, {"lensfun": "lensfunpy non installé"}
    make = meta.get("IFD0:Make") or ""
    model = meta.get("IFD0:Model") or ""
    lens_model = meta.get("ExifIFD:LensModel") or meta.get("Composite:LensID") or ""
    focal = meta.get("ExifIFD:FocalLength")
    db = lensfunpy.Database()
    cams = db.find_cameras(make, model) if make else []
    if not cams:
        return None, {"lensfun": f"boîtier absent de la base ({make} {model})".strip()}
    cam = cams[0]
    lenses = db.find_lenses(cam, None, lens_model) if lens_model else db.find_lenses(cam)
    if not lenses or not focal:
        return None, {"lensfun": f"objectif absent de la base ({lens_model or 'non renseigné'})"}
    H, W = img.shape[:2]
    mod = lensfunpy.Modifier(lenses[0], cam.crop_factor, W, H)
    mod.initialize(float(focal), float(meta.get("ExifIFD:FNumber") or 8.0), 10.0)
    coords = mod.apply_geometry_distortion()
    out = cv2.remap(img, coords[..., 0].astype(np.float32), coords[..., 1].astype(np.float32),
                    cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1), {"lensfun": f"profil {cam.maker} {cam.model} / {lenses[0].model}"}


# ----------------------------------------------------------------- modèle radial
def _maps(W: int, H: int, k: float):
    """Coordonnées source (déformées) pour chaque pixel de sortie (corrigé)."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    R = np.hypot(cx, cy)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xu, yu = (xs - cx) / R, (ys - cy) / R
    f = 1 + k * (xu * xu + yu * yu)
    return (xu * f * R + cx).astype(np.float32), (yu * f * R + cy).astype(np.float32)


def undistort(img: np.ndarray, k: float, extras: tuple = ()):
    """Applique la correction ; sortie sans bord noir. Retourne (image, % du cadre source perdu) ou,
    avec `extras` (masques désignés sur l'original), (image, %, masques transformés à l'identique)."""
    H, W = img.shape[:2]
    mx, my = _maps(W, H, k)
    out = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ex = [cv2.remap(e, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for e in extras]
    valid = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    if k >= 0:  # coussinet corrigé : la sortie déborde la source → recadrage du plus grand rectangle valide
        from .correct import _largest_valid_rect
        x, y, w, h = _largest_valid_rect(valid, W / H)
        out = out[y:y + h, x:x + w]
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LANCZOS4)
        ex = [cv2.resize(e[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_NEAREST) for e in ex]
        perte = 100 * (1 - (w * h) / (W * H))
    else:       # barillet : la sortie n'échantillonne que l'intérieur de la source
        sx = (mx.max() - mx.min()) / (W - 1)
        sy = (my.max() - my.min()) / (H - 1)
        perte = 100 * (1 - min(1, sx) * min(1, sy))
    if extras:
        return np.clip(out, 0, 1), round(float(perte), 2), ex
    return np.clip(out, 0, 1), round(float(perte), 2)


def _straight_support(gray8: np.ndarray, k: float) -> float:
    H, W = gray8.shape
    mx, my = _maps(W, H, k)
    g = cv2.remap(gray8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    diag = np.hypot(H, W)
    lines = cv2.HoughLinesP(e, 1, np.pi / 720, threshold=60, minLineLength=int(0.06 * diag), maxLineGap=4)
    if lines is None:
        return 0.0
    l = np.asarray(lines).reshape(-1, 4).astype(np.float64)
    return float(np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1]).sum())


def estimate_k(img: np.ndarray, max_dim: int = 1400) -> dict:
    g = to_uint8(img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    H, W = g.shape
    s = min(1.0, max_dim / max(H, W))
    g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else g
    base = _straight_support(g, 0.0)
    grid = np.round(np.arange(-0.30, 0.1501, 0.025), 3)
    scores = {float(k): _straight_support(g, float(k)) for k in grid}
    k0 = max(scores, key=scores.get)
    for k in np.round(np.arange(k0 - 0.02, k0 + 0.0201, 0.005), 3):
        scores[float(k)] = _straight_support(g, float(k))
    kb = max(scores, key=scores.get)
    gain = scores[kb] / base - 1 if base > 0 else 0.0
    return {"k": float(kb), "gain_lignes_droites": round(gain, 3), "support_initial": round(base, 1),
            "support_corrige": round(scores[kb], 1)}


def series_k(estimates: list[dict]) -> float | None:
    """Coefficient commun à une série (même boîtier, même focale) : médiane pondérée par le gain des
    estimations nettes. Plus stable qu'une estimation par photo, et cohérent (§4 « cohérence »)."""
    good = [(e["k"], e["gain_lignes_droites"]) for e in estimates if e and e["gain_lignes_droites"] >= GAIN_MIN]
    if len(good) < 2:
        return None
    ks, ws = np.array([g[0] for g in good]), np.array([g[1] for g in good])
    o = np.argsort(ks)
    cw = np.cumsum(ws[o])
    return float(ks[o][np.searchsorted(cw, cw[-1] / 2)])


def correct(img: np.ndarray, meta: dict | None = None, k_impose: float | None = None,
            extras: tuple = ()) -> tuple:
    """Retourne (image, rapport) ou (image, rapport, masques) si `extras`."""
    res = _correct(img, meta, k_impose, tuple(extras))
    return res if extras else res[:2]


def _correct(img, meta, k_impose, extras):
    rep: dict = {"actions": [], "a_reprendre": []}
    f35 = (meta or {}).get("ExifIFD:FocalLengthIn35mmFormat")
    if f35:
        rep["focale_35mm"] = {"valeur": f35, "source": "mesuré", "unite": "mm (EXIF)"}
        if float(f35) < FOCALE_MIN_35MM:
            rep["a_reprendre"].append(f"focale {f35} mm équiv. < {FOCALE_MIN_35MM:g} mm : ultra grand-angle, "
                                      "bords étirés (§4 vise 16–24 mm)")
    if not extras:  # le profil Lensfun ne transforme pas (encore) les masques : réservé aux photos sans masque
        out, lf = lensfun_correct(img, meta)
        rep.update(lf)
        if out is not None:
            rep["actions"].append(f"distorsion corrigée ({lf['lensfun']})")
            rep["appliquee"] = True
            return out, rep, []
    est = estimate_k(img)
    rep["estimation"] = est
    if k_impose is not None:
        # k de série : appliqué seulement s'il ne dégrade pas cette photo (lignes droites ≥ sans correction)
        g = to_uint8(img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        s_ = min(1.0, 1400 / max(g.shape))
        g = cv2.resize(g, (int(g.shape[1] * s_), int(g.shape[0] * s_)), interpolation=cv2.INTER_AREA)
        gain = _straight_support(g, k_impose) / max(1e-6, _straight_support(g, 0.0)) - 1
        k, gain_txt = k_impose, f"k de série, lignes droites {gain:+.0%} sur cette photo"
        if gain < 0:
            rep["actions"].append(f"distorsion : k de série {k_impose:+.3f} non appliqué (dégraderait cette photo, {gain:+.0%})")
            rep["appliquee"] = False
            return img, rep, list(extras)
    else:
        k, gain_txt = est["k"], f"lignes droites {est['gain_lignes_droites']:+.0%}"
        if abs(k) < 0.02 or est["gain_lignes_droites"] < GAIN_MIN:
            rep["actions"].append(f"distorsion : aucune correction nette mesurée (k={k:+.3f}, "
                                  f"gain lignes droites {est['gain_lignes_droites']:+.0%})")
            rep["appliquee"] = False
            return img, rep, list(extras)
    if extras:
        out, perte, ex = undistort(img, k, extras)
    else:
        (out, perte), ex = undistort(img, k), []
    rep["perte_cadre_pct"] = perte
    if perte > PERTE_MAX_PCT:
        rep["actions"].append(f"distorsion k={k:+.3f} mesurée mais non corrigée : la correction "
                              f"retirerait {perte:.1f} % du cadre (> {PERTE_MAX_PCT:g} %)")
        rep["a_reprendre"].append("distorsion forte non corrigeable sans perdre une partie de la pièce")
        rep["appliquee"] = False
        return img, rep, list(extras)
    rep["actions"].append(f"distorsion corrigée (k={k:+.3f}, {gain_txt}, cadre perdu {perte:.1f} %)")
    rep["appliquee"] = True
    rep["k"] = k
    return out, rep, ex
