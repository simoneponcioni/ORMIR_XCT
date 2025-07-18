from .img_types import ImageType, SkeletalSite
from .converters_rescale import ImageConverter
from pathlib import Path
from typing import Union
import SimpleITK as sitk
from ..autocontour.autocontour import Autocontour

# flake8: noqa: E501

class ImageInfo:
    def __init__(self, image_type: ImageType, skeletal_site: SkeletalSite):
        self.image_type = image_type
        self.skeletal_site = skeletal_site


class IMG_XCT:  # looking for a better name
    """Extended XCT Image class based on sitk.Image"""

    def __init__(self,
                 image_path: Union[str, Path],
                 fmt, # outputPixelType
                 image_type: ImageType,
                 skeletal_site: SkeletalSite,
                 image_xct_params: dict = None
                 ):

        self.image_path = Path(image_path)
        self.fmt = fmt
        self.imageinfo = ImageInfo(image_type, skeletal_site)
        self.autocontour = Autocontour()
        self.image_converter = ImageConverter(
            mu_scaling=image_xct_params['mu_scaling'],
            mu_water=image_xct_params['mu_water'],
            rescale_slope=image_xct_params['rescale_slope'],
            rescale_intercept=image_xct_params['rescale_intercept']
        )

        self._image = self.read_image()

        # Initialize segmentations (now these are class attributes)
        self.periosteal = None
        self.endosteal = None
        
    def read_image(self) -> sitk.Image:
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file {self.image_path} does not exist.")
        
        _tmp_img = sitk.ReadImage(str(self.image_path), self.fmt)
        if self.imageinfo.image_type != ImageType.BMD:
            img = self.image_converter.convert(
                _tmp_img,
                from_type=self.imageinfo.image_type,
                to_type=ImageType.BMD
            )
        else:
            img = _tmp_img
        return img

    def segment(self, components=1) -> None:
        """Helper function that calls Autocontour methods to generate periosteal and endosteal contours"""
        self.periosteal_contour = self._segment_periosteal(components)
        # self.endosteal_contour = self._segment_endosteal(components)
    
    def _segment_periosteal(self, components) -> list:
        """Generate periosteal segmentation as list of individual component masks
        
        Returns:
            List of sitk.Image masks, one for each component
        """
        if self.periosteal is not None:
            print(f'Periosteal contour already generated')
            return self.periosteal
        
        # Get individual component masks
        component_masks = []
        for i in range(1, components + 1):
            component_mask = self.autocontour.get_periosteal_mask(self._image, i)
            component_masks.append(component_mask)
        
        # Store the result for future use
        self.periosteal = component_masks
        return component_masks
    
    def _segment_endosteal(self, components) -> list:
        """Generate endosteal segmentation as list of individual component masks
        
        Returns:
            List of sitk.Image masks, one for each component
        """
        # Ensure periosteal mask exists
        if self.periosteal is None:
            print('Periosteal contour not available, generating it first.')
            self._segment_periosteal(components)
        
        # Get individual endosteal component masks
        component_endo_masks = []
        for i in range(1, components + 1):
            # Use combined periosteal mask for endosteal calculation
            combined_periosteal = self.periosteal[0]
            for mask in self.periosteal[1:]:
                combined_periosteal = combined_periosteal + mask
                
            component_endo_mask = self.autocontour.get_endosteal_mask(self._image, combined_periosteal, i)
            component_endo_masks.append(component_endo_mask)
        
        return component_endo_masks