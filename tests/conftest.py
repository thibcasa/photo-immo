import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).parent
ICC_DIR = TESTS / "icc_dl"
SAMPLES = TESTS / "samples_dl"


def _fetch(url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True
    r = subprocess.run(["curl", "-sSfL", "--max-time", "120", "-o", str(dst), url])
    return r.returncode == 0 and dst.exists()


@pytest.fixture(scope="session")
def p3_icc() -> bytes:
    p = ICC_DIR / "DisplayP3-v4.icc"
    if not _fetch("https://cdn.jsdelivr.net/gh/saucecontrol/Compact-ICC-Profiles@master/profiles/DisplayP3-v4.icc", p):
        pytest.skip("profil Display P3 indisponible (réseau)")
    return p.read_bytes()


@pytest.fixture(scope="session")
def dng_sample() -> Path:
    p = SAMPLES / "sample.dng"
    if not p.exists():
        pytest.skip("échantillon DNG absent : lancer tests/fetch_samples.sh")
    return p


def room(W=2400, H=1600, seed=0) -> np.ndarray:
    """Scène synthétique : mur clair, 9 montants verticaux, une ligne d'horizon, bruit léger."""
    a = np.full((H, W, 3), 0.72, np.float32)
    a[:, :, 0] += 0.04  # légère dominante chaude
    for x in np.linspace(150, W - 150, 9):
        cv2.line(a, (int(x), 80), (int(x), H - 80), (0.15, 0.15, 0.15), 6)
    cv2.line(a, (0, H // 2), (W, H // 2), (0.3, 0.3, 0.3), 4)
    rng = np.random.default_rng(seed)
    return np.clip(a + rng.normal(0, 0.008, a.shape).astype(np.float32), 0, 1)


@pytest.fixture
def exiftool_tag():
    if shutil.which("exiftool") is None:
        pytest.skip("exiftool absent")

    def tag(path, *extra):
        subprocess.run(["exiftool", "-q", "-overwrite_original",
                        "-GPSLatitude=43.6961", "-GPSLatitudeRef=N", "-GPSLongitude=7.2718", "-GPSLongitudeRef=E",
                        "-Make=Apple", "-Model=iPhone 15 Pro", "-Artist=Proprietaire Test",
                        "-XMP-dc:Description=12 rue de France Nice", "-IPTC:City=Nice", *extra, str(path)], check=True)
    return tag
