import numpy as np
import SimpleITK as sitk

from ormir_xct.util.hildebrand_thickness import calc_structure_thickness_statistics


def compute_bone_params(seg_img, trab_mask):
    """_summary_

    Parameters
    ----------
    seg_img : SimpleITK image
        Full segmentation (cortical + trabecular bone) obtained from IPL seg_gauss.
    trab_mask : SimpleITK Image
        Trabecular ROI mask.

    Returns
    -------
    dict
        Dictionary containing bone microarchitecture results.
    """
    uct_trab = sitk.Mask(seg_img, trab_mask)
    uct_trab_np = sitk.GetArrayFromImage(uct_trab)

    # ----- Tb.Th -----
    TbTh_stats = calc_structure_thickness_statistics(
        uct_trab_np, seg_img.GetSpacing(), 0, oversample=False, skeletonize=False
    )
    TbTh_mean, TbTh_std = TbTh_stats[0], TbTh_stats[1]

    # Thickness image
    # dt = sitk.GetImageFromArray(TbTh_stats[4])
    # dt.CopyInformation(seg_img)
    # sitk.WriteImage(dt, "/Users/michaelkuczynski/Downloads/KULeuvenCarpals/for_arc/002_uCT_XCT2_REG_scaphoid_seg_TbTh.nii")

    # ----- Tb.Sp -----
    # Invert segmentation and mask with trabecular ROI mask so we
    # don't compute Tb.Sp of the background
    uct_trab_inv = 1 - uct_trab
    bone_inv = sitk.Mask(uct_trab_inv, trab_mask)
    bone_inv_np = sitk.GetArrayFromImage(bone_inv)
    TbSp_stats = calc_structure_thickness_statistics(
        bone_inv_np, seg_img.GetSpacing(), 0, oversample=False, skeletonize=False
    )
    TbSp_mean, TbSp_std = TbSp_stats[0], TbSp_stats[1]

    # Thickness image
    # dt = sitk.GetImageFromArray(TbSp_stats[4])
    # dt.CopyInformation(seg_img)
    # sitk.WriteImage(dt, "/Users/michaelkuczynski/Downloads/KULeuvenCarpals/for_arc/002_uCT_XCT2_REG_scaphoid_seg_TbSp.nii")

    # ----- BV/TV -----
    tv = sitk.GetArrayFromImage(trab_mask)
    bv = sitk.GetArrayFromImage(uct_trab)
    bvtv = (bv > 0).sum() / (tv > 0).sum()

    return {
        "Tb.Th (mm)": TbTh_mean,
        "Tb.Th Std (mm)": TbTh_std,
        "Tb.Sp (mm)": TbSp_mean,
        "Tb.Sp Std (mm)": TbSp_std,
        "BV/TV": bvtv,
    }


def testing():
    seg_img = (
        sitk.ReadImage(
            "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE/00_ORIGAIM/NODARATIS/00000183/00001636/C0001592_UNCOMP_DERIVED/C0001592_UNCOMP_FULLSEG.mhd",
            sitk.sitkUInt8,
        )
        > 0
    )
    trab_mask = (
        sitk.ReadImage(
            "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE/00_ORIGAIM/NODARATIS/00000183/00001636/C0001592_UNCOMP_DERIVED/C0001592_UNCOMP_ENDO.mhd",
            sitk.sitkUInt8,
        )
        > 0
    )
    bone_params = compute_bone_params(seg_img, trab_mask)
    print(bone_params)


if __name__ == "__main__":
    testing()
