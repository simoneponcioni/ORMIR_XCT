from pathlib import Path

from ormir_xct.util.img_types import ImageType, SkeletalSite
from ormir_xct.util.img_xct import IMG_XCT as img_xct
import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np


def main():
    joint_seg_path = Path("ORMIR_XCT/examples/images") / "GRAY_JOINT.nii"

    # Scanner- and calibration-dependent parameters
    img_params = {    
        "mu_scaling": 8192,
        "mu_water": 0.2409,
        "rescale_slope": 1603.51904,
        "rescale_intercept": -391.209015,
    }

    # Initialize IMG_XCT with required parameters
    gray_img = img_xct(
        image_path=joint_seg_path,
        fmt=sitk.sitkFloat32,
        image_type=ImageType.HU,
        skeletal_site=SkeletalSite.HAND,
        image_xct_params=img_params,
    )

    # Now you can use the methods
    # Components = how many bones are present in the image
    # (e.g., prox and dist)
    # TODO: now prx + dst are summed, it would be better to distinguish them
    gray_img.segment(components=2)

    # Access the segmentation results
    periosteal_mask = gray_img.periosteal_contour
    # endosteal_mask = gray_img.endosteal_contour

    return periosteal_mask  # , endosteal_mask


if __name__ == "__main__":
    # periosteal_mask, endosteal_mask = main()
    periosteal_mask = main()
    dst_peri_np = sitk.GetArrayFromImage(periosteal_mask[0])
    prx_peri_np = sitk.GetArrayFromImage(periosteal_mask[1])
    
    dst_peri_np = np.where(dst_peri_np > 0, 1, 0)
    prx_peri_np = np.where(prx_peri_np > 0, 2, 0)
    mask = dst_peri_np + prx_peri_np
    plt.figure()
    plt.imshow(mask[mask.shape[0] // 2, :, :], cmap='gray')
    plt.show()
