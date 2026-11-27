"""Ligne de commande :
  python -m rv2 selfcheck                  → vérifie que l'environnement permet le MODE A
  python -m rv2 models                     → télécharge les modèles de détection (visages, objets)
  python -m rv2 diagnose FICHIER...        → diagnostic étiqueté (JSON)
  python -m rv2 run fiche.json             → traitement complet + RAPPORT.md (§9)
  python -m rv2 export SRC DST [qualité]   → export conforme seul (rotation → sRGB/ICC → sans EXIF/GPS/XMP)
  python -m rv2 video SRC DST [--mention TEXTE --de S --a S] [--sans-stabilisation]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def selfcheck() -> int:
    ok = True
    rows = []

    def chk(name, fn):
        nonlocal ok
        try:
            rows.append((name, "OK", fn()))
        except Exception as exc:
            ok = False
            rows.append((name, "ÉCHEC", str(exc)))

    import importlib
    for mod, attr in [("PIL", "__version__"), ("numpy", "__version__"), ("cv2", "__version__"),
                      ("rawpy", "__version__"), ("pillow_heif", "__version__")]:
        chk(mod, lambda m=mod, a=attr: getattr(importlib.import_module(m), a))
    for binary, args in [("exiftool", ["-ver"]), ("ffmpeg", ["-version"]), ("tesseract", ["--version"])]:
        chk(binary, lambda b=binary, a=args: subprocess.run([b, *a], capture_output=True, text=True, check=True)
            .stdout.splitlines()[0] if shutil.which(b) else (_ for _ in ()).throw(RuntimeError("absent")))
    chk("ffmpeg vid.stab + libx264", lambda: (lambda f, e: "OK" if "vidstabdetect" in f and "libx264" in e else
                                                (_ for _ in ()).throw(RuntimeError("filtre/encodeur manquant")))(
        subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout,
        subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout))
    chk("tesseract langue fra", lambda: "fra" if "fra" in subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                                                                         text=True).stdout else (_ for _ in ()).throw(RuntimeError("fra absent")))
    from .export import _font_path
    chk("police mention", _font_path)
    chk("lensfun (profils d'objectifs)", lambda: __import__("lensfunpy").__version__)
    from .detect import MODELS, SOURCES
    for n in SOURCES:
        chk(f"modèle {n}", lambda n=n: f"{(MODELS / n).stat().st_size // 1024} Ko" if (MODELS / n).exists()
            else (_ for _ in ()).throw(RuntimeError("absent : python -m rv2 models")))
    for name, status, detail in rows:
        print(f"{status:6} {name:28} {detail}")
    print("\nMODE A disponible" if ok else "\nMODE A INCOMPLET → exécuter ./install.sh, sinon MODE B (prescription)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rv2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    sub.add_parser("models")
    d = sub.add_parser("diagnose")
    d.add_argument("fichiers", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("fiche")
    e = sub.add_parser("export")
    e.add_argument("src")
    e.add_argument("dst")
    e.add_argument("qualite", nargs="?", type=int, default=90)
    v = sub.add_parser("video")
    v.add_argument("src")
    v.add_argument("dst")
    v.add_argument("--mention")
    v.add_argument("--de", type=float, action="append", default=[])
    v.add_argument("--a", type=float, action="append", default=[])
    v.add_argument("--sans-stabilisation", action="store_true")
    v.add_argument("--rotation", type=float, default=0.0)
    a = ap.parse_args(argv)

    if a.cmd == "selfcheck":
        return selfcheck()
    if a.cmd == "models":
        from .detect import fetch_models
        print(fetch_models())
        return 0
    if a.cmd == "diagnose":
        from .diagnose import diagnose
        from .io import load_image, read_metadata
        for f in a.fichiers:
            img, info = load_image(f)
            print(json.dumps({"fichier": f, "lecture": info, "diagnostic": diagnose(img, info, read_metadata(f))},
                             ensure_ascii=False, indent=2, default=str))
        return 0
    if a.cmd == "run":
        from .pipeline import run_job
        res = run_job(a.fiche)
        print(Path(res["sortie"]) / "RAPPORT.md")
        return 0 if all(p.get("statut") == "✅" for p in res["photos"]) else 2
    if a.cmd == "export":
        from .export_simple import export
        print(json.dumps(export(a.src, a.dst, a.qualite), ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "video":
        from .video import process_video
        wins = list(zip(a.de, a.a)) or None
        print(json.dumps(process_video(a.src, a.dst, stabilize=not a.sans_stabilisation, rotate_deg=a.rotation,
                                       mention=a.mention, mention_windows=wins), ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
