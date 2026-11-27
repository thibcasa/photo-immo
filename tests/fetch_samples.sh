#!/usr/bin/env bash
# Échantillons de test non versionnés (DNG public raw.pixls.us ~48 Mo, profils ICC compacts).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p samples_dl icc_dl
[ -s samples_dl/sample.dng ] || curl -sSfL --max-time 300 -o samples_dl/sample.dng \
  "https://raw.pixls.us/getfile.php/1022/nice/Google%20-%20Pixel%203a%20-%2012bit%20(4:3).dng"
[ -s icc_dl/DisplayP3-v4.icc ] || curl -sSfL -o icc_dl/DisplayP3-v4.icc \
  "https://cdn.jsdelivr.net/gh/saucecontrol/Compact-ICC-Profiles@master/profiles/DisplayP3-v4.icc"
echo "OK"
