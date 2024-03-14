import torch
import argparse

from utils import get_device, image_to_grid, save_image
from unet import UNet
from ilvr import DDPMWithILVR


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single_ref", "denoising_process"],
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ref_idx", type=int, required=True)
    parser.add_argument("--scale_factor", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=False)

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
    model = DDPMWithILVR(model=net, img_size=args.IMG_SIZE, device=DEVICE)
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    if args.MODE == "single_ref":
        gen_image = model.sample_using_single_ref(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            batch_size=args.BATCH_SIZE,
        )
        gen_grid = image_to_grid(gen_image, n_cols=int((args.BATCH_SIZE + 1) ** 0.5))
        save_image(gen_grid, save_path=args.SAVE_PATH)
    elif args.MODE == "denoising_process":
        model.vis_ilvr(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            scale_factor=args.SCALE_FACTOR,
            batch_size=args.BATCH_SIZE,
            save_path=args.SAVE_PATH,
        )


if __name__ == "__main__":
    main()
