"""Export conforme prompt v2.1 : rotation EXIF -> sRGB + ICC sRGB -> sans EXIF/GPS/XMP.

Même usage que la version d'origine (conservée dans legacy/) :
    python3 export_immo.py SOURCE DESTINATION.jpg [QUALITE]
Corrections : HEIC, RAW/DNG, TIFF 16 bits, CMJN avec profil, refus d'écraser l'original,
contrôle mesuré du fichier écrit. Voir rv2/export_simple.py.
"""
import json
import sys

from rv2.export_simple import export

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    q = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    print(json.dumps(export(sys.argv[1], sys.argv[2], q), ensure_ascii=False, indent=2))
