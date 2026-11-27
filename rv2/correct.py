"""EXÉCUTION des retouches VERT (§6.3), dans l'ordre :
verticales/perspective → exposition/fusion → balance des blancs → suppressions → bruit/netteté
→ floutage adresse. Toutes les opérations sont globales (aucun masque local flou) : pas de halo
possible par construction, sauf `fuse_brackets` (Mertens) dont le résultat doit être audité."""
from __future__ import annotations

import cv2
import numpy as np

from .diagnose import histogram, linear_to_srgb, luminance, measure_verticals, srgb_to_linear, _neutral_mask
from .io import from_uint8, to_uint8

ROTATION_MAX_DEG = 3.0       # §4 : au-delà → À REPRENDRE
PERTE_CADRE_MAX_PCT = 15.0   # seuil technique de l'outil (≈ perte d'une rotation de 3° à ratio constant) :
                             # au-delà, la correction rogne une partie de la pièce → À REPRENDRE (§4)
KEYSTONE_MAX_DEG = 8.0       # seuil technique : convergence plus forte = plongée/contre-plongée marquée


# ----------------------------------------------------------------- géométrie
def _largest_valid_rect(valid: np.ndarray, aspect: float) -> tuple[int, int, int, int]:
    """Plus grand rectangle centré, au même ratio, entièrement dans la zone valide."""
    H, W = valid.shape
    cy, cx = H / 2, W / 2
    integ = cv2.integral(valid.astype(np.uint8))
    lo, hi = 0.3, 1.0
    for _ in range(30):
        s = (lo + hi) / 2
        w = min(W, W * s)
        h = w / aspect
        if h > H:
            h, w = H * s, H * s * aspect
        x0, y0 = int(np.ceil(cx - w / 2)), int(np.ceil(cy - h / 2))
        x1, y1 = int(np.floor(cx + w / 2)), int(np.floor(cy + h / 2))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        area = (x1 - x0) * (y1 - y0)
        inside = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        if area > 0 and inside == area:
            lo = s
        else:
            hi = s
    s = lo
    w = W * s
    h = w / aspect
    if h > H:
        h, w = H * s, H * s * aspect
    x0, y0 = int(np.ceil(cx - w / 2)), int(np.ceil(cy - h / 2))
    return x0, y0, int(np.floor(w)) - 1, int(np.floor(h)) - 1


def _warp_and_crop(img: np.ndarray, Mx: np.ndarray, perspective: bool,
                   extras: tuple = ()) -> tuple[np.ndarray, float, list]:
    """Applique la transformation, recadre sans bord noir ; applique EXACTEMENT la même géométrie
    aux masques `extras` (zones de suppression/floutage désignées sur l'original)."""
    H, W = img.shape[:2]
    ones = np.ones((H, W), np.uint8) * 255
    warp = (lambda a, fl, bv=0: cv2.warpPerspective(a, Mx, (W, H), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)) \
        if perspective else \
        (lambda a, fl, bv=0: cv2.warpAffine(a, Mx, (W, H), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=bv))
    out = warp(img, cv2.INTER_LANCZOS4)
    valid = warp(ones, cv2.INTER_NEAREST) == 255
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    x, y, w, h = _largest_valid_rect(valid, W / H)
    cropped = out[y:y + h, x:x + w]
    ex_out = [warp(e, cv2.INTER_NEAREST)[y:y + h, x:x + w] for e in extras]
    perte = 100.0 * (1 - (w * h) / (W * H))
    return np.clip(cropped, 0, 1), perte, ex_out


def rotate(img: np.ndarray, angle_deg: float, extras: tuple = ()):
    """Rotation (sens trigonométrique OpenCV) puis recadrage sans bord noir.
    Retourne (image, % cadre perdu) ou (image, %, masques) si `extras`."""
    H, W = img.shape[:2]
    Mx = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    out, perte, ex = _warp_and_crop(img, Mx, False, extras)
    return (out, perte, ex) if extras else (out, perte)


def keystone(img: np.ndarray, slope_deg_per_px: float, extras: tuple = ()):
    """Redresse des verticales convergentes (angle(x) = slope·(x − W/2)) par homographie."""
    H, W = img.shape[:2]
    pts_src, pts_dst = [], []
    for xd in (0.0, float(W)):
        th = np.radians(slope_deg_per_px * (xd - W / 2))
        t = np.tan(th) * (H / 2)
        pts_src += [(xd - t, 0.0), (xd + t, float(H))]
        pts_dst += [(xd, 0.0), (xd, float(H))]
    Mx = cv2.getPerspectiveTransform(np.float32(pts_dst), np.float32(pts_src))
    # Mx envoie l'image redressée vers la source ; warpPerspective attend source → destination
    out, perte, ex = _warp_and_crop(img, np.linalg.inv(Mx), True, extras)
    return (out, perte, ex) if extras else (out, perte)


def straighten(img: np.ndarray, extras: tuple = ()):
    """Verticales : roulis puis perspective, rapport avant/après [mesuré].

    Règles (§4/§5) : la correction n'est APPLIQUÉE que si (1) la détection est fiable (lignes
    nombreuses et concordantes) et (2) elle reste dans les limites : rotation ≤ 3°, convergence
    ≤ KEYSTONE_MAX_DEG, cadre perdu ≤ PERTE_CADRE_MAX_PCT. Sinon la géométrie d'origine est conservée
    (un recadrage ne doit jamais masquer une partie de la pièce) et la photo est « à reprendre ».
    Retourne (image, rapport) ou (image, rapport, masques transformés) si `extras`."""
    extras_in = tuple(extras)
    before = measure_verticals(img)
    rep = {"avant": {k: v for k, v in before.items() if not k.startswith("_")}, "actions": [],
           "a_reprendre": [], "perte_cadre_pct": 0.0, "appliquee": False}

    def keep_original(msg: str | None = None):
        if msg:
            rep["actions"].append(msg)
        rep["apres"] = rep["avant"]
        rep["perte_cadre_pct"] = 0.0
        return (img, rep, list(extras_in)) if extras_in else (img, rep)

    if before["inclinaison_deg"]["valeur"] is None or not before.get("fiable"):
        return keep_original("verticales non mesurables de façon fiable (gros plan, peu de lignes droites) : "
                             "aucune correction automatique — contrôle visuel")
    tilt = before["_tilt"]
    if abs(tilt) > ROTATION_MAX_DEG:
        rep["a_reprendre"].append(f"inclinaison {abs(tilt):.1f}° > {ROTATION_MAX_DEG}° (§4)")
        return keep_original(f"rotation {-tilt:+.2f}° nécessaire mais > {ROTATION_MAX_DEG}° : non appliquée")
    out, perte, extras = img, 0.0, list(extras_in)
    actions = []
    if abs(tilt) > 0.1:
        # une ligne d'angle +θ (penche vers la droite en descendant) se redresse par une rotation
        # de −θ dans le repère OpenCV (y vers le bas)
        if extras:
            out, perte, extras = rotate(out, -tilt, tuple(extras))
        else:
            out, perte = rotate(out, -tilt)
        actions.append(f"rotation {-tilt:+.2f}°")
    mid = measure_verticals(out)
    if mid["inclinaison_deg"]["valeur"] is not None and mid.get("fiable"):
        conv = abs(mid["convergence_bords_deg"]["valeur"])
        if conv > KEYSTONE_MAX_DEG:
            rep["a_reprendre"].append(f"convergence {conv:.1f}° : plongée/contre-plongée marquée (§4)")
            return keep_original(f"perspective {conv:.1f}° trop forte pour être corrigée sans déformer : non appliquée")
        if conv > 0.15:
            if extras:
                out2, p2, extras = keystone(out, mid["_slope"], tuple(extras))
            else:
                out2, p2 = keystone(out, mid["_slope"])
            out = out2
            perte = 100 * (1 - (1 - perte / 100) * (1 - p2 / 100))
            actions.append(f"perspective : convergence {mid['convergence_bords_deg']['valeur']:+.2f}° corrigée")
    if perte > PERTE_CADRE_MAX_PCT:
        rep["a_reprendre"].append(f"redressement qui rognerait {perte:.1f}% du cadre (> {PERTE_CADRE_MAX_PCT}%) (§4)")
        return keep_original(f"correction ({', '.join(actions)}) non appliquée : elle rognerait {perte:.1f}% de la pièce")
    rep["actions"] = actions
    rep["appliquee"] = bool(actions)
    rep["perte_cadre_pct"] = round(perte, 2)
    after = measure_verticals(out)
    rep["apres"] = {k: v for k, v in after.items() if not k.startswith("_")}
    return (out, rep, extras) if extras_in else (out, rep)


# ----------------------------------------------------------------- exposition
def _shoulder(lin: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Compression douce des hautes lumières au-delà du genou : aucune valeur > 1, pas d'écrêtage dur."""
    over = lin > knee
    out = lin.copy()
    span = 1 - knee
    out[over] = knee + span * (1 - np.exp(-(lin[over] - knee) / span))
    return out


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    lin = srgb_to_linear(img) * (2.0 ** ev)
    return linear_to_srgb(_shoulder(lin))


EXPO_MEDIANE_MIN = 0.46   # en dessous : sous-exposée → on remonte vers cette cible
BRULES_RECUP_PCT = 1.0    # au-dessus : hautes lumières brûlées → on peut baisser (max −0,5 EV)


def auto_exposure(img: np.ndarray, target_median: float = EXPO_MEDIANE_MIN, max_ev: float = 1.5) -> tuple[np.ndarray, dict]:
    """Asymétrique (constaté sur photo réelle : viser une médiane fixe ASSOMBRISSAIT un séjour lumineux
    bien exposé) :
      - sous-exposée (médiane < cible) → remontée jusqu'à la cible, sans ajouter > 0,5 pt de brûlés ;
      - brûlés > 1 % → baisse jusqu'à ≤ 1 % de brûlés, limitée à −0,5 EV ;
      - sinon : aucune correction (l'aspect lumineux et aéré est conservé)."""
    base = histogram(img)["brules_pct"]["valeur"]
    med0 = float(np.median(luminance(img)))
    if med0 >= target_median:
        if base <= BRULES_RECUP_PCT:
            return img, {"ev": 0.0}
        ev = 0.0
        while ev > -0.5 and histogram(exposure(img, ev - 0.05))["brules_pct"]["valeur"] > BRULES_RECUP_PCT:
            ev -= 0.05
        ev -= 0.05
        return exposure(img, ev), {"ev": round(ev, 2)}
    lo, hi = 0.0, max_ev
    for _ in range(25):
        mid = (lo + hi) / 2
        med = float(np.median(luminance(exposure(img, mid))))
        lo, hi = (mid, hi) if med < target_median else (lo, mid)
    ev = (lo + hi) / 2
    while ev > 0:
        if histogram(exposure(img, ev))["brules_pct"]["valeur"] <= base + 0.5:
            break
        ev -= 0.05
    out = exposure(img, ev)
    return out, {"ev": round(ev, 2)}


def tone(img: np.ndarray, shadows: float = 0.0, highlights: float = 0.0, contrast: float = 0.0) -> np.ndarray:
    """Courbe GLOBALE sur la luminance (ombres +, hautes lumières −, contraste en S) ; teinte préservée."""
    y = np.clip(luminance(img), 1e-4, 1)
    y2 = y + shadows * 0.6 * y * (1 - y) ** 3 - highlights * 0.6 * y ** 3 * (1 - y)
    if contrast:
        y2 = y2 + contrast * 0.5 * (y2 - 0.5) * (1 - np.abs(2 * y2 - 1))
    y2 = np.clip(y2, 0, 1)
    out = img * (y2 / y)[..., None]
    return np.clip(out, 0, 1)


BRACKET_DECALAGE_MAX_PCT = 2.0  # seuil technique : au-delà, ce n'est pas la même prise de vue (trépied bougé)


def fuse_brackets(images: list[np.ndarray], report: dict | None = None) -> np.ndarray:
    """Fusion d'expositions de la MÊME prise de vue (Mertens) — VERT, « window pull ».
    Alignement MTB (insensible à l'exposition) de chaque vue sur la première : décalages [mesuré].
    NB : dans OpenCV 5 (Python), AlignMTB.process() ne renvoie rien ; on applique donc
    calculateShift + shiftMat explicitement."""
    imgs8 = [to_uint8(i) for i in images]
    mtb = cv2.createAlignMTB()
    ref = cv2.cvtColor(imgs8[0], cv2.COLOR_RGB2GRAY)
    aligned, shifts = [imgs8[0]], []
    for im in imgs8[1:]:
        s = tuple(int(v) for v in mtb.calculateShift(ref, cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)))
        shifts.append(s)
        aligned.append(mtb.shiftMat(im, s))
    W = imgs8[0].shape[1]
    if report is not None:
        report["decalages_px"] = shifts
        report["meme_prise"] = all(max(abs(x), abs(y)) / W * 100 <= BRACKET_DECALAGE_MAX_PCT for x, y in shifts)
    fused = cv2.createMergeMertens().process([np.ascontiguousarray(i) for i in aligned])
    fused = np.asarray(fused, np.float32)
    if fused.max() > 1.5:  # certaines versions renvoient l'échelle 0–255
        fused /= 255.0
    return np.clip(fused, 0, 1)


# ----------------------------------------------------------------- balance des blancs
WB_AUTO_FORCE = 0.35      # sans mire : correction partielle (la teinte des matériaux est un invariant §3)
WB_AUTO_GAIN_MAX = 0.15   # gains limités à ±15 % en automatique


def white_balance(img: np.ndarray, method: str = "grayworld", strength: float = WB_AUTO_FORCE,
                  patch: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    lin = srgb_to_linear(img)
    if method == "patch":
        if patch is None:
            raise ValueError("méthode 'patch' : zone neutre (x, y, w, h) obligatoire")
        x, y, w, h = patch
        ref = lin[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0)
        strength = 1.0
    else:
        ref = lin[_neutral_mask(img)].mean(axis=0)
    gains = ref[1] / np.maximum(ref, 1e-6)
    gains = 1 + strength * (gains - 1)
    if method != "patch":
        gains = np.clip(gains, 1 - WB_AUTO_GAIN_MAX, 1 + WB_AUTO_GAIN_MAX)
    out = linear_to_srgb(lin * gains)
    return out, {"methode": method, "force": strength, "gains_RGB": [round(float(g), 3) for g in gains]}


# ----------------------------------------------------------------- suppressions / floutage
def remove_small(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reconstruction CLASSIQUE (Telea, non générative) d'une petite zone. La classe dépend de la
    surface [mesuré] et d'une déclaration humaine « ne touche aucun élément structurel »."""
    m8 = (mask > 0).astype(np.uint8) * 255
    i8 = to_uint8(img)
    rec = cv2.inpaint(i8, m8, 5, cv2.INPAINT_TELEA)
    out = np.clip(img + from_uint8(rec) - from_uint8(i8), 0, 1)
    return out, {"zone_reconstruite_pct": round(float((m8 > 0).mean() * 100), 3), "outil": "OpenCV inpaint Telea (non IA)"}


def blur_regions(img: np.ndarray, rects: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, dict]:
    """Floutage irréversible (pixellisation + flou) : numéros de rue, plaques, boîtes aux lettres, visages."""
    out = img.copy()
    H, W = img.shape[:2]
    for x, y, w, h in rects:
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        roi = out[y:y + h, x:x + w]
        k = max(4, min(w, h) // 6)
        small = cv2.resize(roi, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), k / 2)
    area = sum(min(w, W - x) * min(h, H - y) for x, y, w, h in rects) / (W * H) * 100
    return out, {"zones": len(rects), "surface_pct": round(area, 3)}


def blacken_polygon(img: np.ndarray, pts: list[tuple[int, int]]) -> np.ndarray:
    """Écran de TV allumé → noir (VERT, §5)."""
    out = img.copy()
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    out[m > 0] = 0.02
    return out


def changed_area(before: np.ndarray, after: np.ndarray, thr: float = 3 / 255) -> float:
    """% de pixels modifiés entre deux états de même géométrie [mesuré]."""
    return round(float((np.abs(after - before).max(axis=2) > thr).mean() * 100), 3)


# ----------------------------------------------------------------- bruit / netteté
def denoise(img: np.ndarray, h: float = 3.0) -> np.ndarray:
    i8 = to_uint8(img)
    d = cv2.fastNlMeansDenoisingColored(i8, None, h, h, 7, 21)
    return np.clip(img + from_uint8(d) - from_uint8(i8), 0, 1)


def sharpen(img: np.ndarray, amount: float = 0.4, sigma: float = 1.0, threshold: float = 0.01) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - blur
    detail[np.abs(detail) < threshold] = 0
    return np.clip(img + amount * detail, 0, 1)
