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
    """Save image paths to yaml file for hydra config with specific structure:
    - list of image IDs under grayscale_filenames
    - mapping of IDs to full paths under folder_id

    Args:
        image_paths (List[Path]): list of paths to grayscale images in data_dir
        outpath (Path): path to save yaml file

    Returns:
        None
    """
    # Extract IDs from filenames (part before underscore)
    image_ids = [path.name.split("_")[0] for path in image_paths]

    # Create mapping of IDs to full paths
    id_to_path = {path.name.split("_")[0]: str(path) for path in image_paths}

    yaml_dict = {"images": {"grayscale_filenames": image_ids, "folder_id": id_to_path}}

    with open(outpath, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    return None


def save_ids_to_txt(image_paths: List[Path], outpath: Path) -> None:
    """Save image IDs to text file, one per line

    Args:
        image_paths (List[Path]): list of paths to grayscale images
        outpath (Path): path to save txt file

    Returns:
        None
    """
    image_ids = [path.name.split("_")[0] for path in image_paths]
    with open(outpath, "w") as f:
        f.write("\n".join(image_ids))


def main():
    data_dir = Path("absolute/path/to/data")
    yaml_outpath = Path(__file__).parent / "config_validation.yaml"
    txt_outpath = Path(__file__).parent / "filenames.txt"

    image_paths = get_grayscale_image_paths(data_dir)
    save_as_yaml(image_paths, yaml_outpath)
    save_ids_to_txt(image_paths, txt_outpath)
    print("YAML and TXT files created")
    return None


if __name__ == "__main__":
    main()
