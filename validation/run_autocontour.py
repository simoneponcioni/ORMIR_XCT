from pathlib import Path

import SimpleITK as sitk
from image_reader import ImageReader

from ormir_xct.autocontour.autocontour import autocontour
from ormir_xct.util.segmentation_evaluation import (
    calculate_dice_and_jaccard,
    hausdorff_sitk,
)

from time import time


# flake8: noqa: E501


def run_autocontour(image_path: Path, trabmask_path: Path, cortmask_path: Path):
    print("Read images")
    reader = ImageReader(image_path)
    gray_img, mu_scaling, mu_water, rescale_slope, rescale_intercept = (
        reader.read_image()
    )
    trabmask = ImageReader(trabmask_path).read_image()[0]
    cortmask = ImageReader(cortmask_path).read_image()[0]
    ipl_mask = trabmask + cortmask

    print("Run autocontour")
    start_time = time()
    peri_mask, endo_mask, ormir_mask = autocontour(
        gray_img, mu_scaling, rescale_slope, rescale_intercept
    )
    elapsed_time = time() - start_time

    resampled_ormir = sitk.Resample(
        ormir_mask, ipl_mask, interpolator=sitk.sitkNearestNeighbor
    )

    # Cast both images to the same pixel type
    ipl_mask = sitk.Cast(ipl_mask, sitk.sitkInt8)
    resampled_ormir = sitk.Cast(resampled_ormir, sitk.sitkInt8)

    ormir_np = sitk.GetArrayFromImage(resampled_ormir)
    ipl_np = sitk.GetArrayFromImage(ipl_mask)

    print("Calculate dice, jaccard, hausdorff")
    dice, jaccard = calculate_dice_and_jaccard(ipl_np, ormir_np)
    hausdorff = hausdorff_sitk(ipl_mask, resampled_ormir)

    print("DICE: ", dice)
    print("Jaccard: ", jaccard)
    print("Mean Hausdorff Distance: ", hausdorff[0])
    print("Maximum Hausdorff Distance: ", hausdorff[1])
    return dice, jaccard, hausdorff, elapsed_time
