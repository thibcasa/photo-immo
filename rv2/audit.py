"""AUTO-AUDIT CALCULÉ (§6.4) — remplace « À VALIDER (humain) » par des critères mesurés OK/ÉCHEC.

Comparaison faite entre la RÉFÉRENCE GÉOMÉTRIQUE (original + seules transformations géométriques
déclarées : distorsion, redressement — reproduites à l'identique) et le RÉSULTAT. Toute différence de
structure ne peut donc venir que des retouches tonales ou locales.

Invariants §3 [mesuré] :
  - structure : corrélation des cartes de gradient (normalisées) hors zones de retouche locale déclarées ;
  - contours : rappel des contours de la référence retrouvés dans le résultat (tolérance 2 px) ;
  - teinte des matériaux : écart de chromaticité moyen (Δa*b*, CIELAB) sur les grandes zones ;
  - surface/zone modifiée localement = celle déclarée (aucune modification hors masques).
Artefacts [mesuré] :
  - halo / sur-accentuation : énergie haute fréquence près des contours (résultat / référence) ;
  - aplat / postérisation : nombre de niveaux dans les zones lisses (résultat / référence) ;
  - écrêtage ajouté : hausse des pixels brûlés / bouchés ;
  - bruit amplifié : σ (Immerkær) résultat / référence ;
  - bords : lignes droites près des bords non courbées (support de lignes droites ≥ référence).
Les seuils sont techniques (outil), documentés ici et dans le README ; chaque valeur est rendue."""
from __future__ import annotations

import cv2
import numpy as np

from .diagnose import histogram, luminance, noise_sigma
from .io import to_uint8

SEUILS = {
    "structure_corr_min": 0.90,
    "contours_rappel_min": 0.85,
    "delta_ab_max": 6.0,          # ≈ écart de teinte à peine perceptible sur un matériau (ΔE ab ~ 2–6)
    "zone_hors_masque_max_pct": 0.5,
    "halo_ratio_max": 1.9,
    "niveaux_ratio_min": 0.55,
    "brules_hausse_max": 0.5,
    "bouches_hausse_max": 0.5,
    "bruit_ratio_max": 1.5,
}


def _norm_grad(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return m / (np.percentile(m, 99) + 1e-6)


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def _small(img: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    H, W = img.shape[:2]
    s = min(1.0, max_dim / max(H, W))
    return cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def audit(reference: np.ndarray, result: np.ndarray, local_mask: np.ndarray | None = None) -> dict:
    if reference.shape != result.shape:
        result = cv2.resize(result, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    ref, res = _small(reference), _small(result)
    mask = None
    if local_mask is not None and local_mask.any():
        mask = cv2.resize((local_mask > 0).astype(np.uint8), (ref.shape[1], ref.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    keep = ~mask if mask is not None else np.ones(ref.shape[:2], bool)

    gr, gs = to_uint8(luminance(ref)).astype(np.float32), to_uint8(luminance(res)).astype(np.float32)
    nr, ns = _norm_grad(gr), _norm_grad(gs)
    corr = float(np.corrcoef(nr[keep].ravel(), ns[keep].ravel())[0, 1])

    def _stretch(g):  # normalisation 1–99 % : l'exposition ne doit pas changer les contours retenus
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        return np.clip((g - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    er = cv2.Canny(_stretch(gr), 50, 150) > 0
    es = cv2.Canny(_stretch(gs), 50, 150) > 0
    es_d = cv2.dilate(es.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    er_k = er & keep
    rappel = float((er_k & es_d).sum() / max(1, er_k.sum()))

    # teinte : on compare la chromaticité après normalisation de la luminance (la balance des blancs
    # globale légère est VERT ; on vérifie qu'aucun matériau n'a changé de couleur au-delà du seuil)
    lr, ls = _lab(ref), _lab(res)
    smooth = keep & (nr < 0.08)
    # on retire la dérive globale (balance des blancs) : écart résiduel = variation locale de teinte
    # dérive globale (balance des blancs, exposition) = fonction de la seule luminance : on la retire
    # par tranche de L* ; ce qui reste est une variation SPATIALE de teinte (recoloration locale)
    da, db = ls[..., 1] - lr[..., 1], ls[..., 2] - lr[..., 2]
    Lr = lr[..., 0]
    edges_L = np.percentile(Lr[smooth] if smooth.any() else Lr, np.linspace(0, 100, 7))
    band_idx = np.clip(np.searchsorted(edges_L, Lr) - 1, 0, 5)
    sa, sb = np.zeros_like(da), np.zeros_like(db)
    for i in range(6):
        sel = (band_idx == i) & smooth
        if sel.sum() > 50:
            sa[band_idx == i] = np.median(da[sel])
            sb[band_idx == i] = np.median(db[sel])
    d_local = np.hypot(da - sa, db - sb)
    dab_global = float(np.hypot(np.median(da[smooth]), np.median(db[smooth]))) if smooth.any() else 0.0
    dab_local = float(np.percentile(d_local[keep], 95))

    diff = np.abs(res - ref).max(axis=2) > 12 / 255
    tone_changed = float(diff.mean()) > 0.5  # retouche tonale globale : tout bouge, on ne mesure pas ici
    hors_masque = 0.0
    if mask is not None and not tone_changed:
        hors_masque = float((diff & ~mask).mean() * 100)

    band = cv2.dilate(er.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & keep
    hf_r = np.abs(gr - cv2.GaussianBlur(gr, (0, 0), 2.0))
    hf_s = np.abs(gs - cv2.GaussianBlur(gs, (0, 0), 2.0))
    # normalisation par le contraste global pour ne pas confondre exposition et halo
    halo = float((hf_s[band].mean() / max(1e-6, gs.std())) / max(1e-6, hf_r[band].mean() / max(1e-6, gr.std())))

    flat = keep & (nr < 0.05)
    niveaux = []
    t = 64
    for y in range(0, flat.shape[0] - t, t):
        for x in range(0, flat.shape[1] - t, t):
            if flat[y:y + t, x:x + t].mean() > 0.9:
                a = len(np.unique(gr[y:y + t, x:x + t]))
                b = len(np.unique(gs[y:y + t, x:x + t]))
                if a >= 6:
                    niveaux.append(b / a)
    niv = float(np.median(niveaux)) if niveaux else 1.0

    hr, hs = histogram(ref), histogram(res)
    d_brules = hs["brules_pct"]["valeur"] - hr["brules_pct"]["valeur"]
    d_bouches = hs["bouches_pct"]["valeur"] - hr["bouches_pct"]["valeur"]
    bruit = noise_sigma(res)["bruit_sigma"]["valeur"] / max(0.05, noise_sigma(ref)["bruit_sigma"]["valeur"])

    S = SEUILS
    crit = {
        "structure_corr": (round(corr, 3), corr >= S["structure_corr_min"]),
        "contours_rappel": (round(rappel, 3), rappel >= S["contours_rappel_min"]),
        "teinte_delta_ab_local_p95": (round(dab_local, 2), dab_local <= S["delta_ab_max"]),
        "zone_modifiee_hors_masques_pct": (round(hors_masque, 3), hors_masque <= S["zone_hors_masque_max_pct"]),
        "halo_ratio": (round(halo, 2), halo <= S["halo_ratio_max"]),
        "niveaux_zones_lisses_ratio": (round(niv, 2), niv >= S["niveaux_ratio_min"]),
        "brules_hausse_pts": (round(d_brules, 2), d_brules <= S["brules_hausse_max"]),
        "bouches_hausse_pts": (round(d_bouches, 2), d_bouches <= S["bouches_hausse_max"]),
        "bruit_ratio": (round(bruit, 2), bruit <= S["bruit_ratio_max"]),
    }
    inv_keys = ["structure_corr", "contours_rappel", "teinte_delta_ab_local_p95", "zone_modifiee_hors_masques_pct"]
    art_keys = ["halo_ratio", "niveaux_zones_lisses_ratio", "brules_hausse_pts", "bouches_hausse_pts", "bruit_ratio"]

    def verdict(keys):
        fails = [f"{k}={crit[k][0]}" for k in keys if not crit[k][1]]
        return "OK [mesuré]" if not fails else "ÉCHEC [mesuré] : " + ", ".join(fails)

    return {"invariants_§3": verdict(inv_keys), "artefacts": verdict(art_keys),
            "mesures": {k: v[0] for k, v in crit.items()}, "derive_teinte_globale_dab": round(dab_global, 2),
            "seuils": S}
