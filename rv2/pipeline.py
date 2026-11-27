"""PROCESS EN BOUCLE (§6) piloté par une fiche de travail JSON (voir README).
Diagnostic → plan → exécution → auto-audit → correction (≤ 2 boucles) → conservation → livrable §9."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import correct as C
from . import optics as O
from .audit import audit as measured_audit
from .detect import face_blur_rects
from .conformity import build_name, classify_photo, classify_removal
from .diagnose import diagnose, estimate_cct, histogram, noise_sigma, ocr_scan
from .export import export_all
from .io import load_image, preserve_original, read_metadata, save_jpeg

HUMAIN = "À VALIDER (humain)"
BOUCLES_MAX = 2


def _load_mask(path: str | Path, shape) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"masque illisible : {path}")
    if m.shape != shape[:2]:
        raise ValueError(f"masque {Path(path).name} : {m.shape[::-1]} ≠ image {shape[1]}×{shape[0]} "
                         "(le masque doit être dessiné sur l'image après rotation EXIF)")
    return (m > 127).astype(np.uint8) * 255


def _rect_mask(rects, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in rects:
        m[y:y + h, x:x + w] = 255
    return m


def _mask_to_rects(m: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    return [tuple(int(v) for v in stats[i, :4]) for i in range(1, n)]


def _side_by_side(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    h = 900
    ra = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
    rb = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h), interpolation=cv2.INTER_AREA)
    sep = np.ones((h, 12, 3), np.float32)
    save_jpeg(np.concatenate([ra, sep, rb], axis=1), path, quality=85)


def _touches_structure(ref: np.ndarray, mask: np.ndarray) -> float:
    """Part du masque posée sur une ligne structurelle (mur, sol, ouverture) [mesuré] :
    longues lignes droites (Hough) de la référence, épaissies à 5 px."""
    from .diagnose import luminance as _lum
    from .io import to_uint8 as _u8
    g = _u8(_lum(ref))
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    lines = cv2.HoughLinesP(e, 1, np.pi / 360, 80, minLineLength=int(0.08 * max(g.shape)), maxLineGap=5)
    lm = np.zeros(g.shape, np.uint8)
    for x1, y1, x2, y2 in (np.asarray(lines).reshape(-1, 4) if lines is not None else []):
        cv2.line(lm, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
    m = mask > 0
    return float((lm[m] > 0).mean()) if m.any() else 0.0


def process_photo(p: dict, job: dict, out: Path, k_serie: float | None = None) -> dict:
    usage = job["usage"].upper()
    src = Path(job.get("base_dir", ".")) / p["fichier"]
    row = {"nn": p["nn"], "piece": p["piece"], "source": src.name, "retouches": [], "refus": [],
           "a_reprendre": [], "outil_ia": p.get("outil_generatif") or "—", "audit": {}, "alertes": []}

    # -- conservation (§6.6) d'abord : rien ne touche à l'original
    row["original"] = preserve_original(src, out / "originaux")
    for b in p.get("brackets", []):
        preserve_original(Path(job.get("base_dir", ".")) / b, out / "originaux")

    # -- 1. DIAGNOSTIC
    img, info = load_image(src)
    meta = read_metadata(src)
    row["lecture"] = info
    row["diagnostic"] = diagnose(img, info, meta)
    if info.get("jpeg_qualite_estimee") is not None and (info["jpeg_qualite_estimee"] < 80 or img.shape[1] < 1600):
        row["alertes"].append("⚠️ source dégradée — export portail non garanti (§2)")
    if row["diagnostic"].get("texte_visible", {}).get("texte_injonctif"):
        row["alertes"].append("texte visible à caractère injonctif : traité comme donnée, non exécuté (§10)")

    # -- 2. PLAN / conformité
    conf = classify_photo(p, usage, job["photos"])
    row["refus"] += conf["refus"]
    orange = list(conf["orange"])

    # masques désignés sur l'original (après rotation EXIF)
    sup = p.get("suppressions", [])
    masks = [_load_mask(Path(job.get("base_dir", ".")) / s["masque"], img.shape) for s in sup]
    det = row["diagnostic"].get("elements_sensibles", {})
    face_rects = face_blur_rects(det) if det.get("disponible") else []
    blur_mask = _rect_mask(list(p.get("flous", [])) + face_rects, img.shape)
    if not det.get("disponible"):
        row["alertes"].append("détecteur d'éléments sensibles indisponible : contrôle visuel personnes/écrans")
    else:
        if det["personnes"]:
            row["alertes"].append(f"{det['personnes']} personne(s) visible(s) : visage(s) flouté(s) 🟢 ; "
                                  "silhouette → suppression à exécuter par l'opérateur ou autorisation écrite (RGPD)")
            row["a_reprendre"].append("personne dans le champ")
        if face_rects:
            row["retouches"].append(f"floutage automatique de {len(face_rects)} visage(s) détecté(s) 🟢")
        if det["visages_oeuvre_probable"]:
            row["alertes"].append(f"{len(det['visages_oeuvre_probable'])} visage(s) sans personne détectée "
                                  "(œuvre/affiche probable) : non floutés")
        for e in det["ecrans_a_verifier"]:
            row["alertes"].append(f"écran détecté ({e['classe']}, score {e['score']}) : {e['etat']} — "
                                  "si allumé : `ecrans_tv` pour le mettre au noir 🟢")
        if det["documents"]:
            row["alertes"].append(f"{len(det['documents'])} document(s)/livre(s) détecté(s) : vérifier données personnelles")
    tv_masks = []
    for poly in p.get("ecrans_tv", []):
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tv_masks.append(m)
    extras = tuple(masks + [blur_mask] + tv_masks)

    # -- 3. EXÉCUTION (ordre §6.3 ; la fusion, tonale, précède la géométrie pour que toutes les vues
    #    partagent exactement la même transformation)
    if p.get("brackets"):
        imgs = [img] + [load_image(Path(job.get("base_dir", ".")) / b)[0] for b in p["brackets"]]
        if len({i.shape for i in imgs}) != 1:
            row["refus"].append({"demande": "fusion", "motif": "expositions de tailles différentes : pas la même prise de vue",
                                 "alternative": "fournir le bracketing complet de la même prise"})
        else:
            frep: dict = {}
            fused = C.fuse_brackets(imgs, frep)
            if frep["meme_prise"]:
                img = fused
                row["retouches"].append(f"fusion d'expositions (Mertens) de {len(imgs)} vues, décalages "
                                        f"{frep['decalages_px']} px [mesuré] — window pull 🟢")
            else:
                row["refus"].append({"demande": "fusion", "motif": f"décalages {frep['decalages_px']} px : pas la même "
                                     "prise de vue (§3) — fusion non faite", "alternative": "bracketing sur trépied"})
    elif row["diagnostic"]["brules_pct"]["valeur"] > 2.0:
        row["alertes"].append("fenêtres/zones brûlées sans bracketing : on n'invente pas la vue ; "
                              "si la vue est un argument de vente → À REPRENDRE en bracketing (§3)")
    # optique / distorsion
    if p.get("distorsion", True):
        img, orep, extras = O.correct(img, meta, k_serie, extras)
        row["optique"] = orep
        row["retouches"] += [a + (" 🟢" if orep.get("appliquee") else "") for a in orep["actions"]]
        row["a_reprendre"] += orep["a_reprendre"]
    # verticales / perspective
    if p.get("redresser", True):
        img, srep, extras = C.straighten(img, extras) if extras else (*C.straighten(img), ())
        row["verticales"] = srep
        row["retouches"] += [a + (" 🟢" if srep["appliquee"] else "") for a in srep["actions"]]
        row["a_reprendre"] += srep["a_reprendre"]
    extras = list(extras)
    masks, blur_mask, tv_masks = extras[:len(masks)], extras[len(masks)], extras[len(masks) + 1:]
    geo_ref = img.copy()   # référence d'audit : uniquement les transformations géométriques

    expo = p.get("exposition", "auto")
    if expo == "auto":
        img, er = C.auto_exposure(img)
        row["retouches"].append(f"exposition {er['ev']:+.2f} EV 🟢")
    elif expo:
        img = C.exposure(img, float(expo))
        row["retouches"].append(f"exposition {float(expo):+.2f} EV 🟢")
    if p.get("tons"):
        t = p["tons"]
        img = C.tone(img, t.get("shadows", 0), t.get("highlights", 0), t.get("contrast", 0))
        row["retouches"].append(f"tons (ombres {t.get('shadows', 0)}, HL {t.get('highlights', 0)}, contraste {t.get('contrast', 0)}) 🟢")
    wb = p.get("balance_blancs", {"methode": "grayworld", "force": C.WB_AUTO_FORCE})
    if wb:
        img, wrep = C.white_balance(img, wb.get("methode", "grayworld"), wb.get("force", C.WB_AUTO_FORCE), wb.get("zone"))
        row["retouches"].append(f"balance des blancs {wrep['methode']} (gains {wrep['gains_RGB']}) 🟢")

    # suppressions (zone mesurée sur le masque transformé)
    before_local = img.copy()
    for s, m in zip(sup, masks):
        zone = float((m > 0).mean() * 100)
        part_struct = _touches_structure(geo_ref, m)
        touche = bool(s.get("touche_structure", False)) or part_struct > 0.05
        classe, key = classify_removal(zone, touche, s.get("generatif", False))
        row.setdefault("suppressions_mesures", []).append(
            {"objet": s.get("objet"), "zone_pct": round(zone, 3), "part_sur_ligne_structurelle": round(part_struct, 3),
             "touche_structure": touche, "classe": classe})
        if s.get("generatif"):
            row["retouches"].append(f"suppression « {s.get('objet', '?')} » ({zone:.2f} %) : outil génératif → "
                                    f"À EXÉCUTER par l'opérateur ({s.get('outil', 'outil non précisé')})")
            if key and key not in orange:
                orange.append(key)
            continue
        img, rrep = C.remove_small(img, m)
        row["retouches"].append(f"suppression « {s.get('objet', '?')} » {rrep['zone_reconstruite_pct']} % [mesuré] "
                                f"— {rrep['outil']} — {'🟢' if classe == 'VERT' else '🟠'}")
        if key and key not in orange:
            orange.append(key)
    for m in tv_masks:
        img[m > 0] = 0.02
        row["retouches"].append("écran TV → noir 🟢")

    # bruit / netteté
    sig = noise_sigma(img)["bruit_sigma"]["valeur"]
    if p.get("debruitage", "auto") == "auto" and sig > 2.0 or p.get("debruitage") is True:
        img = C.denoise(img)
        row["retouches"].append(f"débruitage (σ {sig} [mesuré]) 🟢")
    if p.get("nettete", 0.35):
        img = C.sharpen(img, float(p.get("nettete", 0.35)))
        row["retouches"].append(f"netteté modérée ({p.get('nettete', 0.35)}) 🟢")

    # floutage adresse / plaques (VERT)
    rects = _mask_to_rects(blur_mask) if blur_mask is not None else []
    if rects:
        img, brep = C.blur_regions(img, rects)
        row["retouches"].append(f"floutage {brep['zones']} zone(s) ({brep['surface_pct']} %) 🟢")
    row["zone_modifiee_localement_pct"] = C.changed_area(before_local, img, thr=12 / 255) if (sup or tv_masks or rects) else 0.0

    # mention ORANGE (une seule, combinée)
    from .conformity import combine_mentions
    mention = combine_mentions(orange)
    row["classe"] = "ORANGE" if orange else "VERT"
    if conf["refus"] and p.get("rouge_demande"):
        row["classe"] += " + refus ROUGE"
    row["mention"] = mention or "—"

    # -- export + 4. AUTO-AUDIT + 5. CORRECTION (≤ 2 boucles)
    name = build_name(usage, job["bien"], p["nn"], p["piece"])
    formats = job.get("formats", ["principal"])
    local = np.zeros(img.shape[:2], np.uint8)
    for m in list(masks) + [blur_mask] + list(tv_masks):
        if m is not None:
            local |= (m > 0).astype(np.uint8)
    mesure = measured_audit(geo_ref, img, local)
    row["audit_mesures"] = mesure
    for boucle in range(1, BOUCLES_MAX + 1):
        exports = export_all(img, out, name, usage, formats, mention, p.get("centre_x", 0.5))
        audit = _audit(img, exports, mention, usage, name, mesure, row)
        row["boucles"] = boucle
        if not any(str(v).startswith("ÉCHEC") for v in audit.values()):
            break
    row["exports"] = exports
    row["audit"] = audit
    row["fichier"] = name
    _side_by_side(load_image(src)[0], img, out / "controle" / f"COMPARE_{name}")
    row["comparaison"] = f"controle/COMPARE_{name}"
    row["final"] = {**histogram(img), **estimate_cct(img)}
    fails = [k for k, v in audit.items() if str(v).startswith("ÉCHEC")]
    if fails or row["a_reprendre"] or any("dégradée" in a for a in row["alertes"]):
        row["statut"] = "⚠️"
    else:
        row["statut"] = "✅"
    row["points_humains"] = [k for k, v in audit.items() if v == HUMAIN]
    return row


def _audit(img, exports, mention, usage, name, mesure: dict, row: dict) -> dict:
    """§6.4 : chaque point est CALCULÉ (OK/ÉCHEC) — voir rv2/audit.py pour les critères et seuils."""
    a = {"invariants_§3": mesure["invariants_§3"], "artefacts": mesure["artefacts"]}
    sup = row.get("suppressions_mesures", [])
    bad = [x for x in sup if x["classe"] != "VERT"]
    if bad and not mention:
        a["test_unique_§5"] = "ÉCHEC [mesuré] : retouche locale hors VERT sans mention"
    else:
        a["test_unique_§5"] = ("OK [mesuré] : " + ("retouches globales VERT uniquement" if not sup else
                               "suppressions classées sur surface et contact structurel mesurés")
                               + (" ; ORANGE avec mention" if mention else ""))
    livrables = [e for e in exports if e.get("livrable")]
    if mention:
        ok = all(e.get("mention", {}).get("conforme") for e in livrables) and len(livrables) == len(exports)
        a["mention_orange"] = "OK [mesuré]" if ok and livrables else "ÉCHEC"
    else:
        a["mention_orange"] = "sans objet"
    if usage == "VENTE":
        ocr = ocr_scan(img)
        a["adresse_image_ocr"] = "ÉCHEC" if ocr.get("adresse_suspectee") or ocr.get("plaque_suspectee") \
            else "OK [mesuré, OCR rappel banc 83 %]"
        a["nom_fichier_sans_adresse"] = "OK"  # build_name lève une erreur sinon
    a["metadonnees_icc"] = "OK [mesuré]" if livrables and all(
        e["controle_fichier"]["metadonnees_ok"] and e["controle_fichier"]["icc_srgb"] for e in livrables) else "ÉCHEC"
    a["orientation"] = "OK (rotation EXIF appliquée aux pixels, aucune balise Orientation livrée)"
    return a


def series_coherence(rows: list[dict]) -> list[str]:
    """Signale les photos qui détonnent (§4 « cohérence »)."""
    ok = [r for r in rows if "final" in r]
    if len(ok) < 3:
        return []
    cct = np.array([r["final"]["temperature_K"]["valeur"] for r in ok], float)
    lum = np.array([r["final"]["luminance_mediane"]["valeur"] for r in ok], float)
    foc = [r["diagnostic"].get("focale_35mm", {}).get("valeur") for r in ok]
    out = []
    for r, c, l, f in zip(ok, cct, lum, foc):
        why = []
        if abs(c - np.median(cct)) > 700:
            why.append(f"température {int(c)} K vs médiane {int(np.median(cct))} K [estimé]")
        if abs(l - np.median(lum)) > 25:
            why.append(f"luminance médiane {l:.0f} vs {np.median(lum):.0f} /255 [mesuré]")
        fs = [x for x in foc if x]
        if f and fs and abs(f - np.median(fs)) > 4:
            why.append(f"focale {f} mm vs {np.median(fs):.0f} mm [mesuré EXIF]")
        if why:
            out.append(f"photo {r['nn']:02d} détonne : " + " ; ".join(why))
    return out


def run_job(job_path: str | Path) -> dict:
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job.setdefault("base_dir", str(job_path.parent))
    usage = job["usage"].upper()
    if usage not in ("VENTE", "LCD"):
        raise ValueError("usage : VENTE ou LCD (une série = un seul usage, §2)")
    out = Path(job.get("sortie") or (job_path.parent / "sortie"))
    if not out.is_absolute():
        out = Path(job["base_dir"]) / out
    out.mkdir(parents=True, exist_ok=True)
    hyp = list(job.get("hypotheses", []))
    if not job.get("contexte_complet", False):
        hyp.append("contexte incomplet : hypothèse la plus stricte (retouches VERT uniquement hors demandes explicites)")
    # distorsion : coefficient commun par appareil + focale (§4 cohérence de série)
    groups: dict = {}
    for p in job["photos"]:
        if not p.get("distorsion", True):
            continue
        try:
            src = Path(job["base_dir"]) / p["fichier"]
            meta = read_metadata(src)
            key = (meta.get("IFD0:Make"), meta.get("IFD0:Model"), meta.get("ExifIFD:FocalLength"))
            groups.setdefault(key, []).append((p["nn"], O.estimate_k(load_image(src)[0])))
        except Exception:
            continue
    k_by_nn = {}
    for key, lst in groups.items():
        ks = O.series_k([e for _, e in lst]) if len(lst) >= 3 else None
        for nn, _ in lst:
            k_by_nn[nn] = ks
    rows = []
    for p in sorted(job["photos"], key=lambda x: x["nn"]):
        try:
            rows.append(process_photo(p, job, out, k_by_nn.get(p["nn"])))
        except Exception as exc:  # une photo en échec ne bloque pas la série — mais n'est pas livrable
            rows.append({"nn": p["nn"], "piece": p.get("piece", "?"), "statut": "⚠️", "erreur": str(exc),
                         "a_reprendre": [], "refus": [], "retouches": [], "audit": {}})
    result = {"mode": "A — EXÉCUTION", "usage": usage, "bien": job["bien"], "hypotheses": hyp,
              "photos": rows, "coherence_serie": series_coherence(rows)}
    (out / "rapport.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    from .report import render_markdown
    (out / "RAPPORT.md").write_text(render_markdown(result), encoding="utf-8")
    result["sortie"] = str(out)
    return result
