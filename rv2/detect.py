"""Éléments sensibles (§5, §6.1) : personnes, visages, écrans (allumés ?), téléphones, documents.

Modèles ONNX (OpenCV 5 ne lit plus que l'ONNX) :
  - YuNet (OpenCV Zoo, licence MIT) — visages ;
  - YOLOv10n (COCO, poids publiés par onnx-community / Ultralytics, licence AGPL-3.0) — personnes, écrans…
Téléchargés par `python -m rv2 models` dans ~/.cache/rv2/models (non versionnés).

Détection = passe globale + tuiles carrées chevauchantes (un écran vu de biais dans une photo verticale
4096 px n'est trouvé qu'en tuile — constaté sur photo réelle), fusion NMS. Résultats [mesuré — modèle],
avec score : un détecteur peut manquer un élément, le rapport le dit."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .io import to_uint8

MODELS = Path(os.environ.get("RV2_MODELS", Path.home() / ".cache" / "rv2" / "models"))
SOURCES = {
    "yunet.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "yolov10n.onnx": "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx",
}
COCO = {0: "personne", 62: "ecran_tv", 63: "ordinateur", 67: "telephone", 73: "livre_document"}
SEUIL = {0: 0.35, 62: 0.25, 63: 0.25, 67: 0.30, 73: 0.30}
ECRAN_ALLUME_SAT = 0.25       # saturation d'un contenu affiché


def fetch_models() -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        p = MODELS / name
        if not (p.exists() and p.stat().st_size > 100_000):
            subprocess.run(["curl", "-sSfL", "--max-time", "300", "-o", str(p), url], check=True)
        out[name] = p.stat().st_size
    return out


def available() -> bool:
    return all((MODELS / n).exists() for n in SOURCES)


_NET = None


def _yolo():
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromONNX(str(MODELS / "yolov10n.onnx"))
    return _NET


def _run_yolo(bgr: np.ndarray, x0: int, y0: int) -> list:
    H, W = bgr.shape[:2]
    s = 640 / max(H, W)
    nh, nw = int(H * s), int(W * s)
    c = np.full((640, 640, 3), 114, np.uint8)
    c[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    net = _yolo()
    net.setInput(cv2.dnn.blobFromImage(c, 1 / 255.0, (640, 640), swapRB=True))
    out = np.asarray(net.forward()).reshape(-1, 6)
    dets = []
    for x1, y1, x2, y2, sc, k in out:
        k = int(k)
        if k in COCO and sc >= SEUIL[k]:
            dets.append([x0 + x1 / s, y0 + y1 / s, x0 + x2 / s, y0 + y2 / s, float(sc), k])
    return dets


def _tiles(H: int, W: int, overlap: float = 0.4, scales=(1.0, 0.6)) -> list[tuple[int, int, int]]:
    """Tuiles carrées chevauchantes (≥ 40 %) à deux échelles : côté = petit côté de l'image, puis 60 %
    (les petits objets — écran vu de biais — n'atteignent le seuil qu'à l'échelle fine : constaté)."""
    out = []
    for sc in scales:
        side = int(min(H, W) * sc)
        if side < 320:
            continue
        nx = 1 if W == side else int(np.ceil((W - side) / (side * (1 - overlap)))) + 1
        ny = 1 if H == side else int(np.ceil((H - side) / (side * (1 - overlap)))) + 1
        for y in np.linspace(0, H - side, ny).astype(int):
            for x in np.linspace(0, W - side, nx).astype(int):
                if not (sc == 1.0 and H == W):
                    out.append((int(x), int(y), side))
    return out


def _nms(dets: list, iou: float = 0.45) -> list:
    if not dets:
        return []
    keep = []
    for k in {d[5] for d in dets}:
        group = [d for d in dets if d[5] == k]
        boxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in group]
        idx = cv2.dnn.NMSBoxes(boxes, [d[4] for d in group], 0.0, iou)
        keep += [group[i] for i in np.asarray(idx).reshape(-1)]
    return keep


def objects(img: np.ndarray) -> list[dict]:
    bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    dets = _run_yolo(bgr, 0, 0)
    for x0, y0, side in _tiles(H, W):
        dets += _run_yolo(bgr[y0:y0 + side, x0:x0 + side], x0, y0)
    res = []
    for x1, y1, x2, y2, sc, k in _nms(dets):
        box = [int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))]
        d = {"classe": COCO[k], "score": round(sc, 2), "boite": box}
        if k in (62, 63, 67):
            d.update(screen_state(img, box))
        res.append(d)
    return res


def screen_state(img: np.ndarray, box) -> dict:
    """État de l'écran [mesuré], sur la boîte entière (un écran vu de biais n'en occupe qu'une partie,
    le reste est le mur : la moyenne ne convient pas — constaté sur photo réelle) :
    - part de pixels sombres et neutres (dalle éteinte) ≥ 20 % → éteint ;
    - part de pixels lumineux ET saturés (contenu affiché) ≥ 20 % → allumé ;
    - sinon indéterminé (contrôle visuel)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2].reshape(-1, 3)
    if roi.size == 0:
        return {"etat": "indéterminé — à vérifier", "allume": None}
    # référence « mur » = médiane d'un anneau autour de la boîte
    w, h = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = max(0, int(x1 - .3 * w)), max(0, int(y1 - .3 * h)), min(W, int(x2 + .3 * w)), min(H, int(y2 + .3 * h))
    ring_mask = np.ones((Y2 - Y1, X2 - X1), bool)
    ring_mask[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    wall = np.median(img[Y1:Y2, X1:X2][ring_mask], axis=0)
    obj = roi[np.abs(roi - wall).max(axis=1) > 0.08]          # pixels qui ne sont pas du mur
    if len(obj) < 0.10 * len(roi):
        return {"etat": "indéterminé — à vérifier", "allume": None, "part_objet": round(len(obj) / len(roi), 3)}
    L = np.array([0.2126, 0.7152, 0.0722])
    lum_o, lum_w = obj @ L, float(wall @ L)
    sat_o = (obj.max(axis=1) - obj.min(axis=1)) / np.maximum(obj.max(axis=1), 1e-4)
    emissive = float((((sat_o > ECRAN_ALLUME_SAT) & (lum_o > 0.20)) | (lum_o > lum_w + 0.15)).mean())
    neutral_dull = float(((lum_o <= lum_w + 0.10) & (sat_o < 0.20)).mean())
    # Asymétrie volontaire : « éteint » seulement sur preuve forte ; sinon « allumé possible » (contrôle).
    # Un écran vu de profil reflète la pièce : constaté qu'un écran éteint peut paraître « coloré ».
    if neutral_dull >= 0.50 and emissive < 0.25:
        etat = "éteint"
    elif emissive >= 0.25:
        etat = "allumé possible — à vérifier"
    else:
        etat = "indéterminé — à vérifier"
    return {"part_emissive": round(emissive, 3), "part_neutre_terne": round(neutral_dull, 3), "etat": etat,
            "allume": False if etat == "éteint" else None}


def faces(img: np.ndarray, score: float = 0.7) -> list[dict]:
    """Visages (YuNet) sur l'image réduite à 1600 px puis en tuiles : boîtes en pixels d'origine."""
    bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    out = []

    def detect(sub, x0, y0, maxdim):
        h, w = sub.shape[:2]
        s = min(1.0, maxdim / max(h, w))
        small = cv2.resize(sub, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        fd = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (small.shape[1], small.shape[0]), score)
        _, f = fd.detect(small)
        for row in (f if f is not None else []):
            x, y, w_, h_ = row[:4] / s
            out.append([x0 + x, y0 + y, x0 + x + w_, y0 + y + h_, float(row[-1]), 0])

    detect(bgr, 0, 0, 1600)
    for x0, y0, side in _tiles(H, W):
        detect(bgr[y0:y0 + side, x0:x0 + side], x0, y0, 1200)
    return [{"classe": "visage", "score": round(d[4], 2),
             "boite": [int(max(0, d[0])), int(max(0, d[1])), int(min(W, d[2])), int(min(H, d[3]))]}
            for d in _nms(out, 0.3)]


def sensitive(img: np.ndarray) -> dict:
    if not available():
        return {"disponible": False, "source": "modèles absents : `python -m rv2 models` — contrôle visuel requis"}
    objs = objects(img)
    fcs = faces(img)
    persons = [o["boite"] for o in objs if o["classe"] == "personne"]
    for f in fcs:
        # visage d'une personne réelle = centre du visage dans la boîte d'une personne détectée ;
        # sinon visage d'une œuvre, affiche, photo encadrée (constaté : Vénus de Botticelli) → pas de floutage auto
        cx, cy = (f["boite"][0] + f["boite"][2]) / 2, (f["boite"][1] + f["boite"][3]) / 2
        f["personne_reelle"] = any(px1 <= cx <= px2 and py1 <= cy <= py2 for px1, py1, px2, py2 in persons)
    return {"disponible": True, "objets": objs, "visages": fcs,
            "personnes": len(persons),
            "visages_a_flouter": [f for f in fcs if f["personne_reelle"]],
            "visages_oeuvre_probable": [f for f in fcs if not f["personne_reelle"]],
            "ecrans_a_verifier": [o for o in objs if "etat" in o and o["etat"] != "éteint"],
            "documents": [o for o in objs if o["classe"] == "livre_document"],
            "source": "mesuré — YOLOv10n (tuiles 2 échelles) + YuNet ; un détecteur peut manquer un élément très petit ou masqué"}


def face_blur_rects(det: dict, pad: float = 0.25) -> list[tuple[int, int, int, int]]:
    rects = []
    for f in det.get("visages_a_flouter", []):
        x1, y1, x2, y2 = f["boite"]
        w, h = x2 - x1, y2 - y1
        rects.append((int(x1 - pad * w), int(y1 - pad * h), int(w * (1 + 2 * pad)), int(h * (1 + 2 * pad))))
    return rects
