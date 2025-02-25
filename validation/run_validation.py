from pathlib import Path
import pandas as pd
from time import time
from run_autocontour import run_autocontour
from hydra.core.config_store import ConfigStore
from config import Config
import hydra
from omegaconf import DictConfig
import pandas as pd

cs = ConfigStore.instance()
cs.store(name="validation_config_schema", node=Config)


def save_dict_to_csv(results_dict, output_dir="validation"):
    """Save results dictionary to CSV file, appending new results to existing file"""
    df_list = []
    for image_id, metrics in results_dict.items():
        metrics["image_id"] = image_id
        df_list.append(metrics)

    df = pd.DataFrame(df_list)

    cols = ["image_id"] + [col for col in df.columns if col != "image_id"]
    df = df[cols]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / "results_autocontour.csv"

    # If file doesn't exist, create it with headers
    if not filename.exists():
        df.to_csv(filename, index=False)
    else:
        # Append without headers
        df.to_csv(filename, mode="a", header=False, index=False)

    return filename


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config_validation",
)
def validation_runner(cfg: DictConfig) -> None:
    image = cfg.images.grayscale_filenames
    impath = Path(cfg.images.folder_id[image])
    # impath = Path(cfg.images.grayscale_filenames)
    trabmask_path = impath.parent / impath.name.replace("UNCOMP", "TRAB_MASK_UNCOMP")
    cortmask_path = impath.parent / impath.name.replace("UNCOMP", "CORT_MASK_UNCOMP")

    results_dict = {}
    dice, jaccard, hausdorff, elapsed_time = run_autocontour(
        impath, trabmask_path, cortmask_path
    )
    im_name = impath.name.split("_")[0]
    res = {
        "dice": dice,
        "jaccard": jaccard,
        "hausdorff_mean": hausdorff[0],
        "hausdorff_max": hausdorff[1],
        "elapsed_time": elapsed_time,
    }
    results_dict[im_name] = res
    save_dict_to_csv(results_dict)
    return None


if __name__ == "__main__":
    validation_runner()
