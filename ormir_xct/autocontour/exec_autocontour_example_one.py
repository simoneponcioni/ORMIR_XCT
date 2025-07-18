import os

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from ormir_xct.autocontour.autocontour import Autocontour
from ormir_xct.util.converters_rescale import ImageConverter
from ormir_xct.util.img_types import ImageType
from ormir_xct.util.segmentation_evaluation import (calculate_dice_and_jaccard,
                                                    hausdorff_sitk)


def autocontour(img):
    """
    Perform autocontouring on the image data. Assumes the image is already in BMD format.

    Args:
        img: The input image data in BMD format.

    Returns:
        A tuple containing the distal mask, proximal mask, and combined mask.
    """
    # Perform autocontouring
    auto_contour = Autocontour()
    prx_mask = auto_contour.get_periosteal_mask(img, 1)
    dst_mask = auto_contour.get_periosteal_mask(img, 2)

    # Create a mask for the entire joint
    mask = prx_mask + dst_mask
    
    mask_np = sitk.GetArrayFromImage(mask)
    plt.figure()
    plt.imshow(mask_np[mask_np.shape[0] // 2, :, :], cmap='gray')
    plt.title("Combined Mask")
    plt.axis('off')
    plt.show()

    return dst_mask, prx_mask, mask


def main():
    joint_seg_path = os.path.join("ORMIR_XCT/examples/images", "GRAY_JOINT.nii")
    joint_seg_ipl_path = os.path.join("ORMIR_XCT/examples/images", "AUTOCONTOUR_IPL.nii")
    output_path = "images"
    print('reading images')
    print(os.getcwd())
    gray_img = sitk.ReadImage(joint_seg_path, sitk.sitkFloat32)
    ipl_mask = sitk.ReadImage(joint_seg_ipl_path, sitk.sitkUInt8)

    # Scanner- and calibration-dependent parameters
    mu_scaling = 8192  # Scanco XCTII
    mu_water = 0.2409
    rescale_slope = 1603.51904
    rescale_intercept = -391.209015

    print("converting...")
    # Initialize the ImageConverter with the scanner parameters
    conv = ImageConverter(mu_scaling, mu_water, rescale_slope, rescale_intercept)

    # Our `gray_img` is in Hounsfield Units (HU)
    # Other compatible types are BMD, SCANCO, or LINEAR_ATTENUATION
    from_type = ImageType.HU

    # The autocontour function requires the image to be in BMD format, convert if needed
    if from_type != ImageType.BMD:
        bmd_img = conv.convert(gray_img, from_type, ImageType.BMD)

    print("performing autocontour...")
    # Call the autocontour function with the converted image
    dst_mask, prx_mask, ormir_mask = autocontour(bmd_img)
    return dst_mask, prx_mask, ormir_mask


if __name__ == "__main__":
    dst_mask, prx_mask, ormir_mask = main()
