from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ormir_xct.util.img_types import ImageType, SkeletalSite
from ormir_xct.util.img_xct import IMG_XCT as img_xct
from time import time


def main():

    start_time = time()
    # joint_seg_path = Path("ORMIR_XCT/examples/images") / "GRAY_JOINT.nii"
    # greyscale_path = Path(
    #     "/home/simoneponcioni/Documents/02_PROJECTS/2025-ormir-xct/99_TMP/autocontour_tests/C0003111_UNCOMP.mha"
    # )
    greyscale_path = Path(
        "/home/simoneponcioni/Documents/02_PROJECTS/2025-ormir-xct/99_TMP/autocontour_tests/00000193/00001702/C0001657_UNCOMP_crop.mhd"
    )

    # Scanner- and calibration-dependent parameters
    # XCTII Bern
    img_params = {
        "mu_scaling": 8192,
        "mu_water": 0.24220,
        "rescale_slope": 1593.59302,
        "rescale_intercept": -390.541992,
    }

    # Initialize IMG_XCT with required parameters
    image_type_s = ImageType.BMD
    gray_img = img_xct(
        image_path=greyscale_path,
        fmt=sitk.sitkInt16,
        image_type=image_type_s,
        skeletal_site=SkeletalSite.RADIUS_TIBIA,
        image_xct_params=img_params,
    )

    # # XCTII Calgary
    # img_params = {
    #     "mu_scaling": 8192,
    #     "mu_water": 0.24090,
    #     "rescale_slope": 1603.51904,
    #     "rescale_intercept": -391.209015,
    # }

    # Settings Calgary
    # gray_img = img_xct(
    #     image_path=joint_seg_path,
    #     fmt=sitk.sitkFloat32,
    #     image_type=ImageType.HU,
    #     skeletal_site=SkeletalSite.HAND,
    #     image_xct_params=img_params,
    # )

    # Now you can use the methods
    # Components = how many bones are present in the image
    # (e.g., prox and dist)
    gray_img.segment(components=1)

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    # Access the segmentation results
    periosteal_mask = gray_img.periosteal_contour[0]
    # endosteal_mask = gray_img.endosteal_contour[0]
    # cortical_mask = gray_img.cortical_mask[0]

    # empty masks if not computed
    endosteal_mask = sitk.Image(periosteal_mask.GetSize(), sitk.sitkUInt8)
    endosteal_mask.CopyInformation(periosteal_mask)
    cortical_mask = sitk.Image(periosteal_mask.GetSize(), sitk.sitkUInt8)
    cortical_mask.CopyInformation(periosteal_mask)

    for mask, name in zip(
        [gray_img._image, periosteal_mask, endosteal_mask, cortical_mask],
        [f"{image_type_s.display_name}", "PERI", "ENDO", "CORT_MASK"],
    ):
        sitk.WriteImage(
            mask,
            str(greyscale_path.parent / f"{greyscale_path.stem}_{name}.mhd"),
        )

        plt.figure()
        plt.imshow(
            sitk.GetArrayFromImage(mask)[mask.GetSize()[2] // 2, :, :],
            cmap="gray",
        )
        plt.colorbar()
        plt.title(f"{name} Mask Slice")
        plt.axis("off")
        plt.savefig(f"{name.lower()}_mask_slice.png")
    return None


if __name__ == "__main__":
    main()
