from typing import Any

from ormir_xct.util.img_types import ImageType
from ormir_xct.util.scanco_rescale import (
    convert_hu_to_bmd, convert_hu_to_linear_attenuation, convert_hu_to_scanco,
    convert_linear_attenuation_to_bmd, convert_linear_attenuation_to_hu,
    convert_linear_attenuation_to_scanco, convert_scanco_to_bmd,
    convert_scanco_to_hu, convert_scanco_to_linear_attenuation)

# flake8: noqa: E501

class ImageConverter:
    """Handles conversion between different image types using scanco_rescale.py functions."""
    
    def __init__(self, mu_scaling: int, mu_water: float, 
                 rescale_slope: float, rescale_intercept: float):
        """
        Initialize the converter with required scaling parameters.
        
        Args:
            mu_scaling: Scaling factor for Scanco units to linear attenuation
            mu_water: Linear attenuation coefficient of water
            rescale_slope: Slope for converting to BMD
            rescale_intercept: Intercept for converting to BMD
        """
        self.mu_scaling = mu_scaling
        self.mu_water = mu_water
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        
        # Initialize conversion map
        self._conversion_map = {}
        self._setup_conversion_functions()
    
    # Conversion methods from SCANCO
    def _convert_scanco_to_linear_attenuation(self, img):
        return convert_scanco_to_linear_attenuation(img, self.mu_scaling)
    
    def _convert_scanco_to_hu(self, img):
        return convert_scanco_to_hu(img, self.mu_scaling, self.mu_water)
    
    def _convert_scanco_to_bmd(self, img):
        return convert_scanco_to_bmd(img, self.mu_scaling, self.rescale_slope, self.rescale_intercept)
    
    # Conversion methods from LINEAR_ATTENUATION
    def _convert_linear_attenuation_to_scanco(self, img):
        return convert_linear_attenuation_to_scanco(img, self.mu_scaling)
    
    def _convert_linear_attenuation_to_hu(self, img):
        return convert_linear_attenuation_to_hu(img, self.mu_water)
    
    def _convert_linear_attenuation_to_bmd(self, img):
        return convert_linear_attenuation_to_bmd(img, self.rescale_slope, self.rescale_intercept)
    
    # Conversion methods from HU
    def _convert_hu_to_scanco(self, img):
        return convert_hu_to_scanco(img, self.mu_water, self.mu_scaling)
    
    def _convert_hu_to_linear_attenuation(self, img):
        return convert_hu_to_linear_attenuation(img, self.mu_water)
    
    def _convert_hu_to_bmd(self, img):
        return convert_hu_to_bmd(img, self.mu_water, self.rescale_slope, self.rescale_intercept)
    
    # Conversion methods from BMD (inverted from other conversions)
    def _convert_bmd_to_linear_attenuation(self, img):
        return (img - self.rescale_intercept) / self.rescale_slope
    
    def _convert_bmd_to_scanco(self, img):
        linear_att = self._convert_bmd_to_linear_attenuation(img)
        return convert_linear_attenuation_to_scanco(linear_att, self.mu_scaling)
    
    def _convert_bmd_to_hu(self, img):
        linear_att = self._convert_bmd_to_linear_attenuation(img)
        return convert_linear_attenuation_to_hu(linear_att, self.mu_water)
    
    def _setup_conversion_functions(self) -> None:
        """Setup all the conversion functions mapping."""
        self._conversion_map = {
            ImageType.SCANCO: {
                ImageType.LINEAR_ATTENUATION: self._convert_scanco_to_linear_attenuation,
                ImageType.HU: self._convert_scanco_to_hu,
                ImageType.BMD: self._convert_scanco_to_bmd
            },
            ImageType.LINEAR_ATTENUATION: {
                ImageType.SCANCO: self._convert_linear_attenuation_to_scanco,
                ImageType.HU: self._convert_linear_attenuation_to_hu,
                ImageType.BMD: self._convert_linear_attenuation_to_bmd
            },
            ImageType.HU: {
                ImageType.SCANCO: self._convert_hu_to_scanco,
                ImageType.LINEAR_ATTENUATION: self._convert_hu_to_linear_attenuation,
                ImageType.BMD: self._convert_hu_to_bmd
            },
            ImageType.BMD: {
                ImageType.LINEAR_ATTENUATION: self._convert_bmd_to_linear_attenuation,
                ImageType.SCANCO: self._convert_bmd_to_scanco,
                ImageType.HU: self._convert_bmd_to_hu
            }
        }
    
    def convert(self, image_data: Any, from_type: ImageType, to_type: ImageType) -> Any:
        """
        Convert image data from one type to another.
        
        Args:
            image_data: The image data to convert
            from_type: The current image type
            to_type: The target image type
            
        Returns:
            The converted image data
            
        Raises:
            ValueError: If no conversion path is found
        """
        if from_type == to_type:
            return image_data
            
        if from_type in self._conversion_map and to_type in self._conversion_map[from_type]:
            conversion_func = self._conversion_map[from_type][to_type]
            return conversion_func(image_data)
            
        # Try to find a conversion path through an intermediate type
        for intermediate_type in ImageType:
            if (from_type in self._conversion_map and 
                intermediate_type in self._conversion_map[from_type] and
                to_type in self._conversion_map[intermediate_type]):
                
                step1 = self._conversion_map[from_type][intermediate_type]
                step2 = self._conversion_map[intermediate_type][to_type]
                
                intermediate_result = step1(image_data)
                final_result = step2(intermediate_result)
                return final_result
                
        raise ValueError(f"No conversion path available from {from_type} to {to_type}")
    