from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ormir_xct.util.img_types import ImageType, SkeletalSite
from ormir_xct.util.img_xct import IMG_XCT as img_xct


def main():
    # joint_seg_path = Path("ORMIR_XCT/examples/images") / "GRAY_JOINT.nii"
    joint_seg_path = "/home/simoneponcioni/Documents/02_PROJECTS/2025-ormir-xct/ORMIR_XCT/examples/images/C0001592_UNCOMP_1-cropped.nii"

    # Scanner- and calibration-dependent parameters
    
    # XCTII Bern
    img_params = {    
        "mu_scaling": 8192,
        "mu_water": 0.24220,
        "rescale_slope": 1593.59302,
        "rescale_intercept": -390.541992,
    }
    
    # # XCTII Calgary
    # img_params = {    
    #     "mu_scaling": 8192,
    #     "mu_water": 0.24090,
    #     "rescale_slope": 1603.51904,
    #     "rescale_intercept": -391.209015,
    # }

    # Initialize IMG_XCT with required parameters
    gray_img = img_xct(
        image_path=joint_seg_path,
        fmt=sitk.sitkFloat32,
        image_type=ImageType.SCANCO,
        skeletal_site=SkeletalSite.RADIUS_TIBIA,
        image_xct_params=img_params,
    )
    # Settings Calgary
    # gray_img = img_xct(
    #     image_path=joint_seg_path,
    #     fmt=sitk.sitkFloat32,
    #     image_type=ImageType.HU,
    #     skeletal_site=SkeletalSite.HAND,
    #     image_xct_params=img_params,
    # )
    
    # TODO: remove after debugging
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(gray_img._image)[gray_img._image.GetSize()[2] // 2, :, :], cmap='gray')
    plt.title("Original Image Slice")
    plt.axis('off')
    plt.savefig("original_image_slice.png")

    # Now you can use the methods
    # Components = how many bones are present in the image
    # (e.g., prox and dist)
    gray_img.segment(components=1)

    # Access the segmentation results
    periosteal_mask = gray_img.periosteal_contour
    endosteal_mask = gray_img.endosteal_contour

    # prox = periosteal_mask[0]
    # dist = periosteal_mask[1]
    # endosteal_mask = endosteal_mask[0]

    # # FOR DEBUGGING: Plot the masks on top of the grey image: greyscale + contours on top with alpha 0.7
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(sitk.GetArrayFromImage(gray_img._image)[gray_img._image.GetSize()[2] // 2, :, :], cmap='gray')
    # plt.contour(sitk.GetArrayFromImage(prox)[prox.GetSize()[2] // 2, :, :], colors='red', alpha=0.7)
    # plt.contour(sitk.GetArrayFromImage(dist)[dist.GetSize()[2] // 2, :, :], colors='red', alpha=0.7)
    # plt.title("Periosteal Contour")
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(sitk.GetArrayFromImage(gray_img._image)[gray_img._image.GetSize()[2] // 2, :, :], cmap='gray')
    # # plt.contour(sitk.GetArrayFromImage(endosteal_mask)[endosteal_mask.GetSize()[2] // 2, :, :], colors='blue', alpha=0.7)
    # plt.title("Endosteal Contour")
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.savefig("segmentation_results.png")
    # # save periosteal mask prox + dist
    # sitk.WriteImage(prox, "periosteal_mask_prox.nii.gz")
    # sitk.WriteImage(dist, "periosteal_mask_dist.nii.gz")
    
    sitk.WriteImage(periosteal_mask[0], "tibia_periosteal_mask_prox.nii.gz")
    return None


if __name__ == "__main__":
    main()
    peri = sitk.GetArrayFromImage(periosteal_mask[0])
    endo = sitk.GetArrayFromImage(endosteal_mask[0])
