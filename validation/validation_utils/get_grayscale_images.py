from pathlib import Path
from typing import List
import yaml


def get_grayscale_image_paths(data_dir: Path) -> List[Path]:
    """From basedir, get all files with '_UNCOMP.AIM' in name, then
    remove files with 'TRAB' or 'CORT' or 'SEG' in filename

    Args:
        data_dir (Path): basepath to search for images

    Returns:
        List[Path]: list of paths to grayscale images in data_dir
    """

    data_dir = Path(data_dir)
    image_paths = list(data_dir.rglob("**/*_UNCOMP.AIM"))
    image_paths = [
        path
        for path in image_paths
        if not any(x in path.name for x in ["TRAB", "CORT", "SEG"])
    ]
    return image_paths


def save_as_yaml(image_paths: List[Path], outpath: Path) -> None:
    """Save image paths to yaml file for hydra config
    (some hydra configs are hard-coded here for simplicity)

    Args:
        image_paths (List[Path]): list of paths to grayscale images in data_dir
        outpath (Path): path to save yaml file

    Returns:
        None
    """
    yaml_dict = {
        "hydra": {"output_subdir": None},
        "images": {"grayscale_filenames": [str(path) for path in image_paths]},
    }

    with open(outpath, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    return None


def main():
    data_dir = Path("add/here/absolute/path/to/data")
    outpath = Path(__file__).parent / "config_validation.yaml"
    image_paths = get_grayscale_image_paths(data_dir)
    save_as_yaml(image_paths, outpath)
    print("YAML file created")
    return None


if __name__ == "__main__":
    main()
