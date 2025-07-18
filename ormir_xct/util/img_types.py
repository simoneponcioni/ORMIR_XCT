from enum import Enum


class ImageType(Enum):
    """Enumeration of supported image types."""
    SCANCO = 1
    LINEAR_ATTENUATION = 2
    HU = 3
    BMD = 4

    @property
    def display_name(self) -> str:
        """Return human-readable name of the image type."""
        names = {
            ImageType.SCANCO: "Scanco",
            ImageType.LINEAR_ATTENUATION: "Linear Attenuation",
            ImageType.HU: "HU",
            ImageType.BMD: "BMD"
        }
        return names[self]

    def __str__(self) -> str:
        return self.display_name

    def is_scanco(self) -> bool:
        return self == ImageType.SCANCO

    def is_linear_attenuation(self) -> bool:
        return self == ImageType.LINEAR_ATTENUATION

    def is_hu(self) -> bool:
        return self == ImageType.HU

    def is_bmd(self) -> bool:
        return self == ImageType.BMD

    def is_valid(self) -> bool:
        return self in ImageType


class SkeletalSite(Enum):
    """Enumeration of skeletal sites."""
    HAND = 1
    KNEE = 2
    RADIUS_TIBIA = 3

    @property
    def display_name(self) -> str:
        """Return human-readable name of the skeletal site."""
        names = {
            SkeletalSite.HAND: "Hand",
            SkeletalSite.KNEE: "Knee",
            SkeletalSite.RADIUS_TIBIA: "Radius/Tibia"
        }
        return names[self]

    def __str__(self) -> str:
        return self.display_name
