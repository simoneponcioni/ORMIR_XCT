import unittest

import numpy as np
import SimpleITK as sitk

from ormir_xct.util.converters_rescale import ImageConverter
from ormir_xct.util.img_types import ImageType


def create_test_image(dimensions, value):
    """
    Creates a SimpleITK image containing a constant value in the given dimensions.
    """
    array = np.full(dimensions, value, dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image

def images_equal(image1, image2):
    """
    Compare two SimpleITK images by converting them to numpy arrays.
    """
    arr1 = sitk.GetArrayFromImage(image1)
    arr2 = sitk.GetArrayFromImage(image2)
    return np.allclose(arr1, arr2)

class TestImageConverter(unittest.TestCase):
    def setUp(self):
        # Use parameters consistent with the tests in script 2.
        self.mu_scaling = 8192
        self.mu_water = 0.5
        self.rescale_slope = 9
        self.rescale_intercept = 1
        
        # Create an instance of the ImageConverter.
        self.converter = ImageConverter(
            mu_scaling=self.mu_scaling,
            mu_water=self.mu_water,
            rescale_slope=self.rescale_slope,
            rescale_intercept=self.rescale_intercept
        )
        # Dimensions for test images.
        self.dim = (10, 10, 10)
    
    def assertImageEqual(self, img1, img2):
        """Helper assertion to compare two images."""
        self.assertTrue(images_equal(img1, img2), "Images do not match.")

    def test_identity_conversion(self):
        """Test that converting an image to its own type returns the original image."""
        # Test for each image type.
        for img_type in ImageType:
            test_img = create_test_image(self.dim, 1234)
            result = self.converter.convert(test_img, img_type, img_type)
            self.assertImageEqual(result, test_img)

    def test_scanco_to_linear_attenuation(self):
        # For SCANCO -> Linear Attenuation:
        # Expected: image_value / mu_scaling. For a test value of 8192, expect 1.
        test_value = 8192
        expected_value = test_value / self.mu_scaling  # 1
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.SCANCO, ImageType.LINEAR_ATTENUATION)
        self.assertImageEqual(result, expected_img)

    def test_scanco_to_hu(self):
        # For SCANCO -> HU:
        # 1. LinearAttenuation = ScancoUnits / mu_scaling -> 8192/8192 = 1.
        # 2. HU = -1000 + (LinearAttenuation * (1000/mu_water))
        #    => -1000 + (1 * (1000/0.5)) = -1000 + 2000 = 1000.
        test_value = 8192
        expected_value = -1000 + ( (test_value / self.mu_scaling) * (1000 / self.mu_water) )
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.SCANCO, ImageType.HU)
        self.assertImageEqual(result, expected_img)

    def test_scanco_to_bmd(self):
        # For SCANCO -> BMD:
        # 1. LinearAttenuation = ScancoUnits / mu_scaling  => 8192/8192 = 1.
        # 2. BMD = LinearAttenuation * rescale_slope + rescale_intercept
        #    => 1 * 9 + 1 = 10.
        test_value = 8192
        expected_value = (test_value / self.mu_scaling) * self.rescale_slope + self.rescale_intercept
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.SCANCO, ImageType.BMD)
        self.assertImageEqual(result, expected_img)

    def test_linear_attenuation_to_scanco(self):
        # For Linear Attenuation -> SCANCO:
        # ScancoUnits = LinearAttenuation * mu_scaling.
        test_value = 1
        expected_value = test_value * self.mu_scaling  # 1*8192 = 8192.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.LINEAR_ATTENUATION, ImageType.SCANCO)
        self.assertImageEqual(result, expected_img)

    def test_linear_attenuation_to_hu(self):
        # For Linear Attenuation -> HU:
        # HU = LinearAttenuation * (1000/mu_water) - 1000.
        test_value = 1
        expected_value = test_value * (1000 / self.mu_water) - 1000  # 1*(2000) - 1000 = 1000.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.LINEAR_ATTENUATION, ImageType.HU)
        self.assertImageEqual(result, expected_img)

    def test_linear_attenuation_to_bmd(self):
        # For Linear Attenuation -> BMD:
        # BMD = LinearAttenuation * rescale_slope + rescale_intercept.
        test_value = 1
        expected_value = test_value * self.rescale_slope + self.rescale_intercept  # 1*9+1 = 10.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.LINEAR_ATTENUATION, ImageType.BMD)
        self.assertImageEqual(result, expected_img)

    def test_hu_to_linear_attenuation(self):
        # For HU -> Linear Attenuation:
        # LinearAttenuation = (HU + 1000) * (mu_water/1000).
        test_value = 1000
        expected_value = (test_value + 1000) * (self.mu_water / 1000)  # (2000*0.5/1000) = 1.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.HU, ImageType.LINEAR_ATTENUATION)
        self.assertImageEqual(result, expected_img)

    def test_hu_to_scanco(self):
        # For HU -> SCANCO:
        # 1. Compute linear attenuation from HU then multiply by mu_scaling.
        #    LinearAttenuation = (HU + 1000) * (mu_water/1000) -> 1.
        #    ScancoUnits = 1 * mu_scaling = 8192.
        test_value = 1000
        expected_value = ( (test_value + 1000) * (self.mu_water / 1000) ) * self.mu_scaling
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.HU, ImageType.SCANCO)
        self.assertImageEqual(result, expected_img)

    def test_hu_to_bmd(self):
        # For HU -> BMD:
        # 1. LinearAttenuation = (HU + 1000) * (mu_water/1000).
        # 2. BMD = LinearAttenuation * rescale_slope + rescale_intercept.
        test_value = 1000
        linear_att = (test_value + 1000) * (self.mu_water / 1000)
        expected_value = linear_att * self.rescale_slope + self.rescale_intercept  # 1*9+1 = 10.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.HU, ImageType.BMD)
        self.assertImageEqual(result, expected_img)

    def test_bmd_to_linear_attenuation(self):
        # For BMD -> Linear Attenuation:
        # LinearAttenuation = (BMD - rescale_intercept)/rescale_slope.
        # For BMD = 10, expected: (10 - 1)/9 = 1.
        test_value = 10
        expected_value = (test_value - self.rescale_intercept) / self.rescale_slope
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.BMD, ImageType.LINEAR_ATTENUATION)
        self.assertImageEqual(result, expected_img)

    def test_bmd_to_scanco(self):
        # For BMD -> SCANCO:
        # First, convert BMD -> Linear Attenuation then multiply by mu_scaling.
        # For BMD = 10, linear = (10 - 1)/9 = 1, then ScancoUnits = 1 * 8192.
        test_value = 10
        linear_att = (test_value - self.rescale_intercept) / self.rescale_slope
        expected_value = linear_att * self.mu_scaling  # 1*8192 = 8192.
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.BMD, ImageType.SCANCO)
        self.assertImageEqual(result, expected_img)

    def test_bmd_to_hu(self):
        # For BMD -> HU:
        # First, convert BMD -> Linear Attenuation then to HU.
        # For BMD = 10, linear = 1, then HU = 1*(1000/mu_water)-1000 = 2000-1000 = 1000.
        test_value = 10
        linear_att = (test_value - self.rescale_intercept) / self.rescale_slope
        expected_value = linear_att * (1000 / self.mu_water) - 1000
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.BMD, ImageType.HU)
        self.assertImageEqual(result, expected_img)

    def test_fallback_conversion_path(self):
        """
        Test that if a direct conversion mapping is removed,
        the converter finds an intermediate conversion path.
        For example, remove the SCANCO -> BMD mapping and let it
        convert SCANCO -> LinearAttenuation -> BMD.
        """
        # Remove the direct SCANCO -> BMD conversion.
        original_mapping = self.converter._conversion_map[ImageType.SCANCO].pop(ImageType.BMD)
        
        test_value = 8192
        # Expected result using intermediate:
        # SCANCO -> Linear: 8192/8192 = 1, then Linear -> BMD: 1*9+1 = 10.
        expected_value = 10
        test_img = create_test_image(self.dim, test_value)
        expected_img = create_test_image(self.dim, expected_value)
        
        result = self.converter.convert(test_img, ImageType.SCANCO, ImageType.BMD)
        self.assertImageEqual(result, expected_img)
        
        # Restore the original direct mapping for further tests.
        self.converter._conversion_map[ImageType.SCANCO][ImageType.BMD] = original_mapping

    def test_no_conversion_path(self):
        """
        Test that if no conversion path exists,
        the converter raises a ValueError.
        We simulate this by clearing the conversion map.
        """
        # Backup the original map.
        original_map = self.converter._conversion_map.copy()
        self.converter._conversion_map.clear()
        
        test_img = create_test_image(self.dim, 1234)
        with self.assertRaises(ValueError):
            self.converter.convert(test_img, ImageType.HU, ImageType.BMD)
        
        # Restore the original conversion map.
        self.converter._conversion_map = original_map

if __name__ == '__main__':
    unittest.main()
