import torch
import argparse

from utils import get_device, image_to_grid, save_image
from unet import UNet
from ilvr import ILVR


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "denoising_process", "interpolation", "coarse_to_fine"],
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ref_image_idx", type=int, required=True)
    parser.add_argument("--scale_factor", type=int, required=True)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    print(f"[ DEVICE: {DEVICE} ]")
    
    net = UNet()
    model = ILVR(model=net, img_size=args.IMG_SIZE, device=DEVICE)
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    if args.MODE == "denoising_process":
        model.vis_denoising_process(
            batch_size=args.BATCH_SIZE, save_path=args.SAVE_PATH,
        )
    else:
        if args.MODE == "normal":
            gen_image = model.sample_using_ilvr(
                data_dir=args.DATA_DIR,
                ref_image_idx=args.REF_IMAGE_IDX,
                scale_factor=args.SCALE_FACTOR,
            )
            gen_grid = image_to_grid(gen_image, n_cols=1)
            gen_grid.show()
            save_image(gen_grid, save_path=args.SAVE_PATH)
        else:
            if args.MODE  == "interpolation":
                gen_image = model.interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                    interpolate_at=args.INTERPOLATE_AT,
                    n_points=args.N_POINTS,
                )
                gen_grid = image_to_grid(gen_image, n_cols=args.N_POINTS + 2)
                save_image(gen_grid, save_path=args.SAVE_PATH)
            elif args.MODE  == "coarse_to_fine":
                gen_image = model.coarse_to_fine_interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                    n_rows=args.N_ROWS,
                    n_points=args.N_POINTS,
                )
                gen_grid = image_to_grid(gen_image, n_cols=args.N_POINTS + 2)
                save_image(gen_grid, save_path=args.SAVE_PATH)


if __name__ == "__main__":
    main()
