"""VIDÉO (§7) via ffmpeg : stabilisation (vid.stab 2 passes), étalonnage identique sur tout le fichier,
rotation de niveau, mention ORANGE pendant toute la durée des plans concernés, suppression des
métadonnées (GPS, date, appareil), contrôle [mesuré] après écriture."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .export import MENTION_CIBLE_RATIO, _font_path
from .io import privacy_tags, read_metadata


def probe(path: str | Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _video_stream(info: dict) -> dict:
    return next(s for s in info["streams"] if s["codec_type"] == "video")


def process_video(src: str | Path, dst: str | Path, *, stabilize: bool = True, rotate_deg: float = 0.0,
                  brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, gamma: float = 1.0,
                  mention: str | None = None, mention_windows: list[tuple[float, float]] | None = None,
                  crf: int = 18) -> dict:
    """mention_windows : liste (début, fin) en secondes = durée COMPLÈTE de chaque plan ORANGE.
    Aucune modification de vitesse, aucun recadrage autre que celui imposé par la rotation (§7)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        raise ValueError("source = destination : refus (l'original doit rester intact)")
    info = probe(src)
    vs = _video_stream(info)
    W, H = int(vs["width"]), int(vs["height"])
    dur_in = float(info["format"]["duration"])
    filters = []
    with tempfile.TemporaryDirectory() as td:
        trf = Path(td) / "transforms.trf"
        if stabilize:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                            f"vidstabdetect=shakiness=5:accuracy=15:result={trf}", "-f", "null", "-"], check=True)
            filters += [f"vidstabtransform=input={trf}:smoothing=15:optzoom=1", "unsharp=5:5:0.3:3:3:0.0"]
        if abs(rotate_deg) > 0.05:
            import math
            a = math.radians(abs(rotate_deg))
            # facteur de zoom pour supprimer les coins noirs à ratio constant
            z = math.cos(a) + max(W / H, H / W) * math.sin(a)
            filters += [f"rotate={math.radians(rotate_deg)}:fillcolor=black",
                        f"scale=iw*{z:.5f}:ih*{z:.5f}", f"crop={W}:{H}"]
        if (brightness, contrast, saturation, gamma) != (0.0, 1.0, 1.0, 1.0):
            filters.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}")
        if mention:
            tf = Path(td) / "mention.txt"
            tf.write_text(mention, encoding="utf-8")
            fs = max(12, int(round(H * MENTION_CIBLE_RATIO * 1.38)))  # taille de police → hauteur ≥ 3,4 % (capitales + jambages)
            wins = mention_windows or [(0.0, dur_in)]
            enable = "+".join(f"between(t\\,{a:.3f}\\,{b:.3f})" for a, b in wins)
            filters.append(f"drawtext=fontfile={_font_path()}:textfile={tf}:fontsize={fs}:fontcolor=white:"
                           f"box=1:boxcolor=black@0.68:boxborderw={int(fs * 0.35)}:x=(w-text_w)/2:"
                           f"y=h-text_h-{int(H * 0.025) + int(fs * 0.35)}:enable='{enable}'")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-map_metadata", "-1", "-map_chapters", "-1",
                "-metadata:s:v:0", "handler_name=", "-metadata:s:a:0", "handler_name=",
                "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
    out_info = probe(dst)
    dur_out = float(out_info["format"]["duration"])
    leaks = privacy_tags(read_metadata(dst))
    return {"source": src.name, "sortie": dst.name, "filtres": filters,
            "duree_entree_s": round(dur_in, 3), "duree_sortie_s": round(dur_out, 3),
            "duree_conservee": abs(dur_out - dur_in) < 0.15,
            "balises_sensibles": leaks, "metadonnees_ok": not leaks,
            "mention": mention, "mention_fenetres": mention_windows,
            "source_mesures": "mesuré (ffprobe + exiftool sur le fichier écrit)"}
