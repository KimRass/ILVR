import torch
from torch import nn
import numpy as np
import imageio
from tqdm import tqdm
from pathlib import Path

from resizeright import downsample_then_upsample
from celeba import CelebADS
from metfaces import MetFacesDS
from utils import image_to_grid, create_dir


class DDPM(nn.Module):
    def linearly_schedule_beta(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def __init__(
        self,
        model,
        img_size,
        device,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = model.to(device)

        self.linearly_schedule_beta()

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step][:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, diffusion_step, rand_noise=None):
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image
        var = 1 - alpha_bar_t
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * rand_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        return self.model(
            noisy_image=noisy_image.to(self.device), diffusion_step=diffusion_step,
        )

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self(
            noisy_image=noisy_image.detach(), diffusion_step=diffusion_step,
        )
        model_mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        model_var = beta_t

        if diffusion_step_idx > 0:
            rand_noise = self.sample_noise(batch_size=noisy_image.size(0))
        else:
            rand_noise = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return model_mean + (model_var ** 0.5) * rand_noise

    @staticmethod
    def _get_frame(x):
        grid = image_to_grid(x, n_cols=int(x.size(0) ** 0.5))
        frame = np.array(grid)
        return frame


class DDPMWithILVR(DDPM):
    def select_and_batchify_ref(self, dataset, data_dir, ref_idx, batch_size):
        if dataset == "celeba":
            ds = CelebADS(
                data_dir=data_dir, split="test", img_size=self.img_size, hflip=False,
            )
        elif dataset == "metfaces":
            ds = MetFacesDS(data_dir=data_dir, img_size=self.img_size, hflip=False)
        return ds[ref_idx][None, ...].to(self.device).repeat(batch_size, 1, 1, 1)

    @torch.inference_mode()
    def refine_latent_var(
        # self, noisy_image, ref, diffusion_step_idx, last_cond_step=None, scale_factor=None,
        self, noisy_image, ref, diffusion_step_idx, last_cond_step, scale_factor=None,
    ):
        """
        "Algorithm 1" line 7 to 9;
        "${x^{\prime}_{t - 1}} \sim p_{\theta}(x^{\prime}_{t - 1} \vert x_{t})$"
        "$y_{t - 1} \sim q(y_{t - 1} \vert y)$"
        "$x_{t - 1} \leftarrow \phi_{N}(y_{t - 1}) + x^{\prime}_{t - 1} - \phi_{N}(x^{\prime}_{t - 1})$"
        """
        less_noisy_image = self.take_denoising_step(noisy_image, diffusion_step_idx=diffusion_step_idx)
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        noisy_ref = self.perform_diffusion_process(
            ori_image=ref,
            diffusion_step=diffusion_step,
        )
        # print(less_noisy_image.shape, noisy_ref.shape)
        # return less_noisy_image + downsample_then_upsample(
        #     noisy_ref, scale_factor=scale_factor,
        # ) - downsample_then_upsample(less_noisy_image, scale_factor=scale_factor)
        # if last_cond_step is None:
        #     last_cond_step = self.batchify_diffusion_steps(
        #         diffusion_step_idx=0, batch_size=noisy_image.size(0),
            # )
        # print(diffusion_step.shape, last_cond_step.shape, noisy_ref.shape, less_noisy_image.shape)
        return torch.where(
            diffusion_step[:, None, None, None].repeat(
                1, self.image_channels, self.img_size, self.img_size,
            ) >= last_cond_step[:, None, None, None].repeat(
                1, self.image_channels, self.img_size, self.img_size,
            ),
            less_noisy_image + downsample_then_upsample(
                noisy_ref, scale_factor=scale_factor,
            ) - downsample_then_upsample(less_noisy_image, scale_factor=scale_factor),
            less_noisy_image,
        )

    def perform_ilvr(
        self,
        noisy_image,
        ref,
        start_diffusion_step_idx,
        scale_factor,
        last_cond_step,
        n_frames=None,
    ):
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            # if diffusion_step_idx >= last_cond_step_idx:
            #     x = self.refine_latent_var(
            #         x,
            #         ref=ref,
            #         diffusion_step_idx=diffusion_step_idx,
            #         scale_factor=scale_factor,
            #     )
            # else:
            #     x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)
            x = self.refine_latent_var(
                noisy_image=x,
                ref=ref,
                diffusion_step_idx=diffusion_step_idx,
                last_cond_step=last_cond_step,
                scale_factor=scale_factor,
            )

            if n_frames is not None and (
                diffusion_step_idx % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(torch.cat([ref[: 1, ...], x], dim=0)))
        return frames if n_frames is not None else x

    def sample_using_single_ref(
        self,
        data_dir,
        ref_idx,
        scale_factor,
        batch_size,
        last_cond_step_idx=0,
        dataset="celeba",
    ):
        rand_noise = self.sample_noise(batch_size=batch_size - 1)
        ref = self.select_and_batchify_ref(
            dataset=dataset,
            data_dir=data_dir,
            ref_idx=ref_idx,
            batch_size=batch_size - 1,
        )
        last_cond_step = self.batchify_diffusion_steps(
            diffusion_step_idx=last_cond_step_idx, batch_size=batch_size - 1,
        )
        gen_image = self.perform_ilvr(
            noisy_image=rand_noise,
            ref=ref,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            last_cond_step=last_cond_step,
            scale_factor=scale_factor,
            n_frames=None,
        )
        return torch.cat([ref[: 1, ...], gen_image], dim=0)

    def sample_using_various_scale_factors(
        self,
        data_dir,
        ref_idx,
        last_cond_step_idx=0,
        dataset="celeba",
    ):
        return self.sample_using_single_ref(
            data_dir=data_dir,
            ref_idx=ref_idx,
            scale_factor=None,
            batch_size=6,
            last_cond_step_idx=last_cond_step_idx,
            dataset=dataset,
        )

    def sample_using_various_cond_range(
        self,
        data_dir,
        ref_idx,
        scale_factor,
        last_cond_step_indices=[125, 250, 375, 500, 625, 750, 875, 1000],
        dataset="celeba",
    ):
        batch_size = len(last_cond_step_indices) + 1
        rand_noise = self.sample_noise(batch_size=batch_size - 1)
        ref = self.select_and_batchify_ref(
            dataset=dataset,
            data_dir=data_dir,
            ref_idx=ref_idx,
            batch_size=batch_size - 1,
        )
        last_cond_step = torch.tensor(last_cond_step_indices, device=self.device)
        gen_image = self.perform_ilvr(
            noisy_image=rand_noise,
            ref=ref,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            last_cond_step=last_cond_step,
            scale_factor=scale_factor,
            n_frames=None,
        )
        return torch.cat([ref[: 1, ...], gen_image], dim=0)

    # def perform_ilvr_using_various_cond_range(
    #     self,
    #     noisy_image,
    #     ref,
    #     scale_factor,
    #     start_diffusion_step_idx=999,
    #     last_cond_step_indices=[125, 250, 375, 500, 625, 750, 875, 1000],
    #     n_frames=None,
    # ):
    #     batch_size = len(last_cond_step_indices)
    #     ref = self.select_and_batchify_ref(
    #         dataset=dataset,
    #         data_dir=data_dir,
    #         ref_idx=ref_idx,
    #         batch_size=batch_size,
    #     )

    #     diffusion_step = self.batchify_diffusion_steps(
    #         diffusion_step_idx=999, batch_size=len(last_cond_step_indices),
    #     )


    #     diffusion_step >= torch.tensor(last_cond_step_indices)

    #     rand_noise = self.sample_noise(batch_size=batch_size - 1)
    #     x = rand_noise
    #     pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
    #     for diffusion_step_idx in pbar:
    #         pbar.set_description("Denoising...")

    #         x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)
    #         diffusion_step = self.batchify_diffusion_steps(
    #             diffusion_step_idx=diffusion_step_idx, batch_size=1,
    #         )
    #         noisy_ref = self.perform_diffusion_process(
    #             ori_image=ref,
    #             diffusion_step=diffusion_step,
    #         )

    #         torch.where(
    #             diffusion_step >= torch.tensor(last_cond_step_indices),
    #             x + downsample_then_upsample(
    #                 noisy_ref, scale_factor=scale_factor,
    #             ) - downsample_then_upsample(x, scale_factor=scale_factor),
    #             x,
    #         )

    #         x = x + downsample_then_upsample(
    #             noisy_ref, scale_factor=scale_factor,
    #         ) - downsample_then_upsample(x, scale_factor=scale_factor)



    def vis_ilvr(
        self,
        data_dir,
        ref_idx,
        scale_factor,
        batch_size,
        save_path,
        dataset="celeba",
        n_frames=100,
    ):
        rand_noise = self.sample_noise(batch_size=batch_size - 1)
        ref = self.select_and_batchify_ref(
            dataset=dataset,
            data_dir=data_dir,
            ref_idx=ref_idx,
            batch_size=batch_size,
        )
        frames = self.perform_ilvr(
            noisy_image=rand_noise,
            ref=ref,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            scale_factor=scale_factor,
            n_frames=n_frames,
        )
        create_dir(save_path)
        imageio.mimsave(Path(save_path).with_suffix(".gif"), frames)
