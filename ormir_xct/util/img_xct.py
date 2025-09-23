from pathlib import Path
from typing import Union

import SimpleITK as sitk

from ..autocontour.autocontour import Autocontour
from .converters_rescale import ImageConverter
from .img_types import ImageType, SkeletalSite
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from validation.image_reader import ImageReader

# flake8: noqa: E501


class ImageInfo:
    def __init__(self, image_type: ImageType, skeletal_site: SkeletalSite):
        self.image_type = image_type
        self.skeletal_site = skeletal_site


class IMG_XCT:  # looking for a better name
    """Extended XCT Image class based on sitk.Image"""

    def __init__(
        self,
        image_path: Union[str, Path],
        fmt,  # outputPixelType
        image_type: ImageType,
        skeletal_site: SkeletalSite,
        image_xct_params: dict = None,
    ):

        self.image_path = Path(image_path)
        self.fmt = fmt
        self.imageinfo = ImageInfo(image_type, skeletal_site)
        self.autocontour = Autocontour()
        if image_xct_params is None:
            print("Reading image parameters from AIM header")
            pass
        else:
            self.image_converter = ImageConverter(
                mu_scaling=image_xct_params["mu_scaling"],
                mu_water=image_xct_params["mu_water"],
                rescale_slope=image_xct_params["rescale_slope"],
                rescale_intercept=image_xct_params["rescale_intercept"],
            )

        self._image = self.read_image()

        # Initialize segmentations (now these are class attributes)
        self.periosteal = None
        self.endosteal = None
        self.cortical_mask = None

    def read_image(self) -> sitk.Image:
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file {self.image_path} does not exist.")

        if "aim" in self.image_path.suffix.lower():
            # open with AIM reader
            reader = ImageReader(self.image_path)
            _tmp_img, scaling, slope, intercept = reader.read_image()
            self.image_converter = ImageConverter(
                mu_scaling=scaling,
                mu_water=0.24220,  #! hardcoded for testing (POS, 22.09.2025)
                rescale_slope=slope,
                rescale_intercept=intercept,
            )
        else:
            _tmp_img = sitk.ReadImage(str(self.image_path), self.fmt)
        if self.imageinfo.image_type != ImageType.BMD:
            print(f"Converting image from {self.imageinfo.image_type} to BMD")
            img = self.image_converter.convert(
                _tmp_img, from_type=self.imageinfo.image_type, to_type=ImageType.BMD
            )
        else:
            img = _tmp_img

        # transpose 2, 1, 0
        # img = sitk.PermuteAxes(img, (2, 1, 0))
        print(img.GetSize())

        #! remove after testing!
        # crop z-axis, only 50 slices
        img = img[:, :, 100:150]

        _PAD = 20
        img = sitk.ConstantPad(img, (_PAD, _PAD, 0), (_PAD, _PAD, 0), 0.0)

        return img

    def segment(self, components=1) -> None:
        """Helper function that calls Autocontour methods to generate periosteal and endosteal contours"""
        self.periosteal_contour = self._segment_periosteal(components)
        self.cortical_mask, self.endosteal_contour = self._segment_endosteal(components)

    def _segment_periosteal(self, components) -> list:
        """Generate periosteal segmentation as list of individual component masks

        Returns:
            List of sitk.Image masks, one for each component
        """
        print(f"Segmenting periosteal contour for {self.image_path.name}")
        if self.periosteal is not None:
            print(f"Periosteal contour already generated")
            return self.periosteal

        # Get individual component masks
        component_masks = []
        segmentations = []
        for i in range(1, components + 1):
            seg, component_mask = self.autocontour.get_periosteal_mask(self._image, i)
            component_masks.append(component_mask)
            segmentations.append(seg)

        # Store the result for future use
        self.periosteal = component_masks
        self.img_segmented = segmentations
        return component_masks

    def _segment_endosteal(self, components) -> list:
        """Generate endosteal segmentation as list of individual component masks

        Returns:
            List of sitk.Image masks, one for each component
        """
        # Ensure periosteal mask exists
        if self.periosteal is None:
            print("Periosteal contour not available, generating it first.")
            self._segment_periosteal(components)

        print(f"Segmenting endosteal contour for {self.image_path.name}")
        # Get individual endosteal component masks
        cort = [None] * components
        cortical_mask = []
        endosteal_contour = []
        for i in range(1, components + 1):
            # Calculate cortical mask from each periosteal component
            cort[i - 1], endosteal_surface = self.autocontour.get_endosteal_mask(
                self._image, self.periosteal[i - 1]
            )
            cortical_mask.append(cort[i - 1])
            endosteal_contour.append(endosteal_surface)

        return cortical_mask, endosteal_contour
