"""Tests des 4 limites levées : distorsion, éléments sensibles, OCR, audit calculé."""
import numpy as np
import cv2
import pytest

from conftest import room
from rv2 import correct as C
from rv2 import optics as O
from rv2.audit import audit
from rv2 import detect as D


# ----------------------------------------------------------------- distorsion
def _distort(base, k):
    H, W = base.shape[:2]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    R = np.hypot(cx, cy)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xd, yd = (xs - cx) / R, (ys - cy) / R
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        f = 1 + k * (xu * xu + yu * yu)
        xu, yu = xd / f, yd / f
    return cv2.remap(base, (xu * R + cx).astype(np.float32), (yu * R + cy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("k", [-0.12, -0.06, 0.05])
def test_distorsion_estimee(k):
    est = O.estimate_k(_distort(room(), k))
    assert abs(est["k"] - k) <= 0.015 and est["gain_lignes_droites"] > 0.3


def test_pas_de_correction_sans_distorsion():
    out, rep = O.correct(room())
    assert not rep["appliquee"] and out is not None


def test_k_de_serie_mediane_ponderee():
    est = [{"k": -0.08, "gain_lignes_droites": 0.9}, {"k": -0.07, "gain_lignes_droites": 0.4},
           {"k": -0.30, "gain_lignes_droites": 0.11}, {"k": 0.0, "gain_lignes_droites": 0.02}]
    assert O.series_k(est) == -0.08


def test_masques_suivent_la_distorsion():
    img = _distort(room(), -0.1)
    m = np.zeros(img.shape[:2], np.uint8)
    m[100:200, 100:300] = 255
    out, perte, (mt,) = O.undistort(img, -0.1, (m,))
    assert mt.shape == m.shape and mt.any() and perte < 15


def test_focale_ultra_grand_angle_a_reprendre():
    _, rep = O.correct(room(), {"ExifIFD:FocalLengthIn35mmFormat": 6, "IFD0:Make": "Arashi Vision",
                                "IFD0:Model": "insta360 x5"})
    assert any("6 mm" in x for x in rep["a_reprendre"])
    assert "absent de la base" in rep["lensfun"]           # aucun profil Insta360 dans Lensfun (constaté)


# ----------------------------------------------------------------- éléments sensibles
def test_etat_ecran():
    wall = np.full((600, 800, 3), (0.80, 0.74, 0.66), np.float32)
    on = wall.copy()
    on[200:400, 300:560] = np.random.default_rng(0).uniform(0.4, 1, (200, 260, 3)) * [0.3, 0.6, 1.0] + 0.1
    off = wall.copy()
    off[200:400, 300:560] = 0.08
    assert D.screen_state(on, [300, 200, 560, 400])["etat"].startswith("allumé")
    assert D.screen_state(off, [300, 200, 560, 400])["etat"] == "éteint"


def test_tuiles_couvrent_l_image():
    tiles = D._tiles(4096, 2304)
    cover = np.zeros((4096, 2304), bool)
    for x, y, s in tiles:
        cover[y:y + s, x:x + s] = True
    assert cover.all() and {s for *_, s in tiles} == {2304, 1382}


def test_visage_flou_seulement_si_personne(monkeypatch):
    monkeypatch.setattr(D, "available", lambda: True)
    monkeypatch.setattr(D, "objects", lambda img: [{"classe": "personne", "score": .9, "boite": [0, 0, 100, 300]}])
    monkeypatch.setattr(D, "faces", lambda img: [{"classe": "visage", "score": .9, "boite": [30, 20, 70, 60]},
                                                 {"classe": "visage", "score": .9, "boite": [400, 20, 440, 60]}])
    d = D.sensitive(np.zeros((500, 500, 3), np.float32))
    assert len(d["visages_a_flouter"]) == 1 and len(d["visages_oeuvre_probable"]) == 1
    assert len(D.face_blur_rects(d)) == 1


@pytest.mark.skipif(not D.available(), reason="modèles absents : python -m rv2 models")
def test_detecteurs_charges():
    d = D.sensitive(room(1200, 800))
    assert d["disponible"] and d["personnes"] == 0


# ----------------------------------------------------------------- OCR
@pytest.mark.slow
def test_ocr_rappel_banc():
    from ocr_bench import run
    r = run(24)
    nouveau = int(r["nouveau"].split("/")[0])
    ancien = int(r["ancien"].split("/")[0])
    assert nouveau >= 18 and nouveau > ancien            # mesuré : 20/24 contre 8/24


# ----------------------------------------------------------------- audit calculé
def _scene():
    a = room(1600, 1066)
    a[700:1000, 200:600] = (0.55, 0.42, 0.30)          # « parquet »
    return a


def test_audit_bonne_retouche_ok():
    img = _scene()
    r, _ = C.auto_exposure(img * 0.8)
    r, _ = C.white_balance(r)
    r = C.sharpen(r, 0.35)
    a = audit(img * 0.8, r)
    assert a["invariants_§3"].startswith("OK") and a["artefacts"].startswith("OK"), a["mesures"]


def test_audit_detecte_les_defauts():
    img = _scene()
    assert audit(img, C.sharpen(img, 4.0, 1.5, 0))["artefacts"].startswith("ÉCHEC")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[700:1000, 200:600, 2] += 20
    assert "teinte" in audit(img, cv2.cvtColor(lab, cv2.COLOR_Lab2RGB))["invariants_§3"]
    eff = img.copy()
    eff[100:900, 500:900] = eff[100:900, 500:900].mean((0, 1))
    assert audit(img, eff)["invariants_§3"].startswith("ÉCHEC")
    m = np.zeros(img.shape[:2], np.uint8)
    m[100:900, 500:900] = 255
    assert audit(img, eff, m)["invariants_§3"].startswith("OK")   # déclaré = accepté


def test_exposition_ne_assombrit_pas_une_photo_lumineuse():
    """Régression constatée sur la série MANDARINE : séjour lumineux assombri par une cible fixe."""
    img = np.clip(room() * 1.25, 0, 0.97)          # lumineuse, peu de brûlés
    out, rep = C.auto_exposure(img)
    assert rep["ev"] == 0.0 and out is img
    sombre, rep2 = C.auto_exposure(room() * 0.35)
    assert rep2["ev"] > 0.5


def test_export_reproductible_sans_date_icc(tmp_path):
    """Constaté : deux exports identiques différaient par la date de création du profil ICC."""
    import time
    from rv2.io import save_jpeg, read_metadata, verify_delivered_file
    img = room(600, 400)
    save_jpeg(img, tmp_path / "a.jpg")
    time.sleep(1.1)
    save_jpeg(img, tmp_path / "b.jpg")
    assert (tmp_path / "a.jpg").read_bytes() == (tmp_path / "b.jpg").read_bytes()
    m = read_metadata(tmp_path / "a.jpg")
    assert str(m.get("ICC-header:ProfileDateTime", "0000")).startswith("0000")
    assert verify_delivered_file(tmp_path / "a.jpg")["icc_srgb"]
