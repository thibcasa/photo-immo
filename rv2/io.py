"""Entrées/sorties : lecture multi-format (JPG, PNG, TIFF 16 bits, HEIC, RAW/DNG),
gestion couleur vers sRGB, métadonnées (exiftool), export JPEG sans EXIF/GPS mais avec ICC sRGB,
conservation des originaux (ORIG_)."""
from __future__ import annotations

import hashlib
import io as _io
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

import pillow_heif

pillow_heif.register_heif_opener()

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr", ".iiq"}
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_ICC = SRGB_PROFILE.tobytes()


class Rv2Error(RuntimeError):
    pass


# ----------------------------------------------------------------- utilitaires
def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def icc_description(icc: bytes | None) -> str | None:
    if not icc:
        return None
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(_io.BytesIO(icc))).strip()
    except Exception:  # profil illisible
        return "illisible"


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def from_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


# ----------------------------------------------------------------- lecture
def load_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Retourne (image float32 HxWx3 sRGB [0,1], infos).

    - RAW/DNG : dématriçage LibRaw (rawpy), balance des blancs boîtier, sans auto-luminosité,
      sortie sRGB 16 bits.
    - Autres : rotation EXIF appliquée, profil ICC converti en sRGB si différent.
    """
    path = Path(path)
    ext = path.suffix.lower()
    info: dict = {"source": path.name, "format": ext.lstrip("."), "icc_source": None,
                  "icc_converti": False, "orientation_appliquee": None, "bits": 8}

    if ext in RAW_EXT:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                    output_color=rawpy.ColorSpace.sRGB, gamma=(2.4, 12.92))
        img = rgb16.astype(np.float32) / 65535.0
        info.update(bits=16, icc_source="sRGB (dématriçage LibRaw)")
        info["largeur"], info["hauteur"] = img.shape[1], img.shape[0]
        return img, info

    if ext in {".tif", ".tiff", ".png"}:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
        if arr is not None and arr.dtype == np.uint16:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=2)
            arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
            img = arr.astype(np.float32) / 65535.0
            info.update(bits=16, icc_source="non lu (16 bits, supposé sRGB)")
            info["largeur"], info["hauteur"] = img.shape[1], img.shape[0]
            return img, info

    im = Image.open(path)
    im.load()
    try:
        info["orientation_appliquee"] = im.getexif().get(0x0112)
    except Exception:
        pass
    info["jpeg_qualite_estimee"] = estimate_jpeg_quality(im)
    icc = im.info.get("icc_profile")
    info["icc_source"] = icc_description(icc)
    im = ImageOps.exif_transpose(im)                       # 1. rotation EXIF -> pixels
    # 2. conversion sRGB AVANT tout changement de mode (un profil CMJN s'applique à des pixels CMJN)
    if icc and info["icc_source"] and "srgb" not in info["icc_source"].lower() and im.mode in ("RGB", "CMYK"):
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(_io.BytesIO(icc)), SRGB_PROFILE,
                                           renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
            info["icc_converti"] = True
        except Exception as exc:
            info["icc_erreur"] = f"conversion ICC impossible ({exc}) : pixels traités comme sRGB"
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
    img = from_uint8(np.asarray(im))
    info["largeur"], info["hauteur"] = img.shape[1], img.shape[0]
    return img, info


# Table de quantification luminance standard (IJG, annexe K de la norme JPEG)
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def estimate_jpeg_quality(im: Image.Image) -> int | None:
    """Qualité JPEG équivalente IJG, déduite de la table de quantification [estimé]."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    table = np.array(q[0], dtype=np.float64)  # comparaison triée : indépendante de l'ordre zigzag
    best, best_err = None, None
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        ref = np.clip(np.floor((_STD_LUMA * scale + 50) / 100), 1, 255)
        err = np.abs(np.sort(ref) - np.sort(table)).sum()
        if best_err is None or err < best_err:
            best, best_err = quality, err
    return best


# ----------------------------------------------------------------- métadonnées
def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def read_metadata(path: str | Path) -> dict:
    if not exiftool_available():
        raise Rv2Error("exiftool absent : impossible de mesurer les métadonnées")
    out = subprocess.run(["exiftool", "-j", "-a", "-G1", "-n", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)[0]


_ALLOWED_GROUPS = re.compile(r"^(File|System|ExifTool|JFIF|ICC.*|APP14|Composite|QuickTime|Track\d+|Main)$")
_SENSITIVE_KEY = re.compile(r"GPS|Location|Latitude|Longitude|Make|Model|Serial|Owner|Artist|Author|Creator|"
                            r"Copyright|Software|DateTimeOriginal|CreateDate|UserComment|Address|City|Comment",
                            re.I)
# clés techniques QuickTime/Composite produites par l'encodeur, non personnelles
_HARMLESS = re.compile(r"(Encoder|CompressorName|HandlerVendorID|HandlerType|HandlerDescription|MajorBrand|"
                       r"CompatibleBrands|MinorVersion|MediaCreateDate|MediaModifyDate|TrackCreateDate|"
                       r"TrackModifyDate|ModifyDate)$")


def privacy_tags(meta: dict) -> dict:
    """Balises qui ne doivent pas subsister dans un fichier livré (EXIF, GPS, XMP, IPTC, MakerNotes,
    ou toute clé de localisation/identité, quel que soit le groupe)."""
    found = {}
    for key, val in meta.items():
        if key == "SourceFile":
            continue
        group, _, name = key.partition(":")
        # propriétés du système de fichiers / de l'outil / du profil couleur : pas des données embarquées
        if group in ("File", "System", "ExifTool") or group.startswith("ICC"):
            continue
        if "Date" in name and str(val).startswith("0000:00:00"):
            continue  # date remise à zéro (conteneur MP4/MOV) : aucune information
        if _HARMLESS.search(name):
            # dates QuickTime : nulles (0000:00:00) après -map_metadata -1 ; sinon = date de tournage
            if "Date" in name and not (isinstance(val, (str, int)) and str(val).startswith(("0000:00:00", "0"))):
                found[key] = val
            continue
        if not _ALLOWED_GROUPS.match(group) or _SENSITIVE_KEY.search(name):
            found[key] = val
    return found


def preserve_original(src: str | Path, dest_dir: str | Path) -> dict:
    """Copie l'original intact sous ORIG_[nom] (métadonnées incluses) et vérifie l'empreinte."""
    src, dest_dir = Path(src), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / f"ORIG_{src.name}"
    h_src = sha256(src)
    if dst.exists():
        if sha256(dst) != h_src:
            raise Rv2Error(f"{dst.name} existe déjà avec un contenu différent : refus d'écraser")
    else:
        shutil.copy2(src, dst)
    h_dst = sha256(dst)
    if h_dst != h_src:
        raise Rv2Error("copie ORIG_ non identique à la source")
    return {"orig": str(dst), "sha256": h_src}


# ----------------------------------------------------------------- export
def save_jpeg(img: np.ndarray, path: str | Path, quality: int = 90) -> Path:
    """JPEG sRGB, profil ICC sRGB embarqué, AUCUNE donnée EXIF/XMP/IPTC écrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(to_uint8(img), "RGB")
    im.save(path, "JPEG", quality=quality, optimize=True, subsampling=0, icc_profile=SRGB_ICC)
    return path


def verify_delivered_file(path: str | Path) -> dict:
    """Contrôle [mesuré] d'un fichier livré : aucune balise sensible, ICC sRGB présent."""
    meta = read_metadata(path)
    leaks = privacy_tags(meta)
    icc = meta.get("ICC_Profile:ProfileDescription") or meta.get("ICC-header:ProfileDescription")
    if icc is None:
        icc = next((v for k, v in meta.items() if k.endswith(":ProfileDescription")), None)
    return {"fichier": Path(path).name, "balises_sensibles": leaks, "metadonnees_ok": not leaks,
            "icc": icc, "icc_srgb": bool(icc and "srgb" in str(icc).lower())}
