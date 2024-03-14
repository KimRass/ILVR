import torch
import argparse
from pathlib import Path
import re
import gc

from utils import get_device, image_to_grid, save_image
from unet import UNet
from ilvr import DDPMWithILVR


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--scale_factor", type=int, required=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ref_idx", type=int, required=True)

    parser.add_argument("--batch_size", type=int, required=False)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single_ref", "denoising_process", "various_scale_factors"],
    )

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def get_sample_num(x, pref):
    match = re.search(pattern=rf"{pref}-\s*(.+)", string=x)
    return int(match.group(1)) if match else -1


def get_max_sample_num(samples_dir, pref):
    stems = [path.stem for path in Path(samples_dir).glob("**/*") if path.is_file()]
    return max([get_sample_num(stem, pref=pref) for stem in stems])


def get_save_path(samples_dir, mode, dataset, ref_idx, scale_factor, suffix):
    pref = f"mode={mode}/dataset={dataset}/ref_idx={ref_idx}-scale_factor={scale_factor}"
    max_sample_num = get_max_sample_num(samples_dir, pref=pref)
    save_stem = f"{pref}-{max_sample_num + 1}"
    return str((Path(samples_dir)/save_stem).with_suffix(suffix))


def main():
    torch.set_printoptions(linewidth=70)

    args = get_args()
    DEVICE = get_device()
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    SAMPLES_DIR = Path(__file__).resolve().parent/"samples"

    net = UNet()
    model = DDPMWithILVR(model=net, img_size=args.IMG_SIZE, device=DEVICE)
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    if args.MODE == "single_ref":
        gen_image = model.sample_using_single_ref(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            batch_size=args.BATCH_SIZE,
            dataset=args.DATASET,
        )
        gen_grid = image_to_grid(gen_image, n_cols=int((args.BATCH_SIZE + 1) ** 0.5))
        save_path = get_save_path(
            samples_dir=SAMPLES_DIR,
            mode=args.MODE,
            dataset=args.DATASET,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            suffix=".jpg",
        )
        save_image(gen_grid, save_path=save_path)
    elif args.MODE == "various_scale_factors":
        gen_image = model.sample_using_various_scale_factors(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            dataset=args.DATASET,
        )
        gen_grid = image_to_grid(gen_image, n_cols=gen_image.size(0))
        save_path = get_save_path(
            samples_dir=SAMPLES_DIR,
            mode=args.MODE,
            dataset=args.DATASET,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            suffix=".jpg",
        )
        save_image(gen_grid, save_path=save_path)
    elif args.MODE == "denoising_process":
        save_path = get_save_path(
            samples_dir=SAMPLES_DIR,
            mode=args.MODE,
            dataset=args.DATASET,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            suffix=".gif",
        )
        model.vis_ilvr(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            batch_size=args.BATCH_SIZE,
            save_path=save_path,
            dataset=args.DATASET,
        )


if __name__ == "__main__":
    main()
