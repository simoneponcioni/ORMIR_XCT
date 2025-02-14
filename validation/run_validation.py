from pathlib import Path

from run_autocontour import run_autocontour


def main():
    impath = Path("/Users/msb/Documents/01_PHD/03_Methods/CLEAN-HFE-ACCURATE/00_ORIGAIM/TIBIA/434_L_90_F/C0003111_UNCOMP.AIM")
    ipl_path = Path("/Users/msb/Documents/01_PHD/03_Methods/CLEAN-HFE-ACCURATE/00_ORIGAIM/TIBIA/434_L_90_F/C0003111_UNCOMP_SEG.AIM")
    dice, jaccard, hausdorff = run_autocontour(impath, ipl_path)
    return None

if __name__ == "__main__":
    main()