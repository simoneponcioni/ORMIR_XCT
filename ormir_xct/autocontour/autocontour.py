import os
import argparse
import SimpleITK as sitk

from ormir_xct.autocontour.AutocontourKnee import AutocontourKnee
from ormir_xct.util.scanco_rescale import convert_scanco_to_hu, convert_scanco_to_bmd


def autocontour(img, mu_scaling, rescale_slope, rescale_intercept):
    print("Converting to BMD units")
    img = convert_scanco_to_bmd(img, mu_scaling, rescale_slope, rescale_intercept)

    auto_contour = AutocontourKnee()
    print("Getting periosteal mask")
    peri_mask = auto_contour.get_periosteal_mask(img, 1)
    print("Getting endosteal mask")
    endo_mask = auto_contour.get_endosteal_mask(img, peri_mask)
    mask = peri_mask + endo_mask
    # print("Writing mask")
    # sitk.WriteImage(mask, "mask_autocontour_tmp.mha")
    return peri_mask, endo_mask, mask


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Image (path + filename)")
    parser.add_argument(
        "mu_water",
        type=float,
        nargs="?",
        default="0.2409",
        help="Linear attenuation of water (default = 0.2409)",
    )
    parser.add_argument(
        "rescale_slope",
        type=float,
        nargs="?",
        default="1603.51904",
        help="Slope to scale to BMD (default = 1603.51904)",
    )
    parser.add_argument(
        "rescale_intercept",
        type=float,
        nargs="?",
        default="-391.209015",
        help="Intercept to scale to BMD (default = -391.209015)",
    )
    args = parser.parse_args()

    image_path = args.image_path
    mu_water = args.mu_water
    rescale_slope = args.rescale_slope
    rescale_intercept = args.rescale_intercept

    # Create a new folder to hold the output images
    image_dir = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    prx_mask_path = os.path.join(image_dir, basename + "_PRX_MASK.nii")
    dst_mask_path = os.path.join(image_dir, basename + "_DST_MASK.nii")
    mask_path = os.path.join(image_dir, basename + "_MASK.nii")

    # Read in images as floats to increase precision
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)

    # Run the autocontour method for each bone
    dst_mask, prx_mask, mask = autocontour(
        image, mu_water, rescale_slope, rescale_intercept
    )

    sitk.WriteImage(mask, mask_path)
    sitk.WriteImage(prx_mask, prx_mask_path)
    sitk.WriteImage(dst_mask, dst_mask_path)


if __name__ == "__main__":
    main()
