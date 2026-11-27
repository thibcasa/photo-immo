"""Export conforme prompt v2.1 : rotation EXIF -> sRGB + ICC sRGB -> sans EXIF/GPS/XMP."""
import io, sys
from PIL import Image, ImageOps, ImageCms
SRGB = ImageCms.createProfile("sRGB")
SRGB_BYTES = ImageCms.ImageCmsProfile(SRGB).tobytes()
def export(src, dst, quality=90):
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)              # 1. rotation appliquée aux pixels
    icc = im.info.get("icc_profile")
    im = im.convert("RGB")
    if icc:                                       # 2. conversion vers sRGB
        im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), SRGB,
                                       renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
    clean = Image.frombytes("RGB", im.size, im.tobytes())   # 3. pixels seuls : aucune métadonnée héritée
    clean.save(dst, "JPEG", quality=quality, icc_profile=SRGB_BYTES, subsampling=0)
if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
