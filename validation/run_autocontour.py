from pathlib import Path

import numpy as np
import SimpleITK as sitk
from image_reader import ImageReader
from matplotlib import pyplot as plt

from ormir_xct.autocontour.autocontour import autocontour
from ormir_xct.util.segmentation_evaluation import (calculate_dice_and_jaccard,
                                                    hausdorff_sitk)


def run_autocontour(image_path: Path, ipl_path: Path):
    reader = ImageReader(image_path)
    ipl_mask = ImageReader(ipl_path).read_image()[0]
    gray_img, scaling, slope, intercept = reader.read_image()
    dst_mask, prx_mask, ormir_mask = autocontour(
    gray_img, mu_water, rescale_slope, rescale_intercept
)
    
    resampled_ormir = sitk.Resample(
    ormir_mask, ipl_mask, interpolator=sitk.sitkNearestNeighbor
)

    ormir_np = sitk.GetArrayFromImage(resampled_ormir)
    dice, jaccard = calculate_dice_and_jaccard(ipl_np, ormir_np)
    hausdorff = hausdorff_sitk(ipl_mask, resampled_ormir)

    print("DICE: ", dice)
    print("Jaccard: ", jaccard)
    print("Mean Hausdorff Distance: ", hausdorff[0])
    print("Maximum Hausdorff Distance: ", hausdorff[1])
    return dice, jaccard, hausdorff
