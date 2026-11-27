"""Export conforme seul (§5 « TOUS USAGES, à l'export ») — successeur de export_immo.py.

Corrige deux défauts mesurés sur la version d'origine :
  1. HEIC (format natif iPhone) : plantage « cannot identify image file » → lecteur HEIF enregistré ;
  2. source = destination : l'original était écrasé (contraire au §6 ORIG_) → refus explicite.
Ajoute : RAW/DNG et TIFF 16 bits, conversion ICC avant tout changement de mode (CMJN),
contrôle [mesuré] du fichier écrit (exiftool), suppression du fichier si le contrôle échoue."""
from __future__ import annotations

from pathlib import Path

from .io import Rv2Error, load_image, save_jpeg, verify_delivered_file


def export(src: str | Path, dst: str | Path, quality: int = 90) -> dict:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and dst.resolve() == src.resolve():
        raise Rv2Error("source = destination : refus d'écraser l'original (§6 : l'original reste intact)")
    if dst.suffix.lower() not in (".jpg", ".jpeg"):
        raise Rv2Error("destination : .jpg attendu (§8)")
    img, info = load_image(src)            # 1. rotation EXIF → pixels ; 2. conversion sRGB
    save_jpeg(img, dst, quality=quality)   # 3. pixels seuls + ICC sRGB, aucune EXIF/GPS/XMP/IPTC/commentaire
    check = verify_delivered_file(dst)     # 4. contrôle [mesuré]
    if not (check["metadonnees_ok"] and check["icc_srgb"]):
        dst.unlink(missing_ok=True)
        raise Rv2Error(f"contrôle après export en échec, fichier supprimé : {check}")
    return {"source": src.name, "sortie": str(dst), "lecture": info, "controle": check}
