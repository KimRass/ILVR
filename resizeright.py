# References:
    # https://github.com/assafshocher/ResizeRight
    # https://github.com/jychoi118/ilvr_adm/blob/main/resizer.py

import torch
import torch.nn.functional as F
import numpy as np
import math
from fractions import Fraction
import os

from utils import get_device, save_image, image_to_grid

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def support_size(size):
    def wrapper(f):
        f.support_size = size
        return f
    return wrapper


@support_size(4)
def cubic(x):
    # absx = np.abs(x)
    absx = np.abs(x.cpu())
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.) * (absx <= 1.) +(
            -0.5 * absx3 + 2.5 * absx2 - 4. * absx + 2.
        ) * ((1. < absx) & (absx <= 2.))
    )


@support_size(4)
def lanczos2(x):
    eps = np.finfo(np.float32).eps
    return ((np.sin(math.pi * x) * np.sin(math.pi * x / 2) + eps) / (
            (math.pi ** 2 * x ** 2 / 2) + eps
        )) * (abs(x) < 2)


@support_size(6)
def lanczos3(x):
    eps = np.finfo(np.float32).eps
    return ((np.sin(math.pi * x) * np.sin(math.pi * x / 3) + eps) / (
            (math.pi ** 2 * x ** 2 / 3) + eps
        )) * (abs(x) < 3)


@support_size(2)
def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


@support_size(1)
def box(x):
    return ((-1 <= x) & (x < 0)) + ((0 <= x) & (x <= 1))


def resize(
        input,
        scale_factors=None,
        out_shape=None,
        interp_mode=cubic,
        support_size=None,
        antialiasing=True,
        by_convs=False,
        scale_tolerance=None,
        max_numerator=10,
        pad_mode='constant',
    ):
    # get properties of the input tensor
    in_shape, n_dims = input.shape, input.ndim

    # fw stands for framework that can be either numpy or torch,
    # determined by the input type
    fw = np if type(input) is np.ndarray else torch
    eps = fw.finfo(fw.float32).eps
    device = input.device if fw is torch else None

    # set missing scale factors or output shapem one according to another,
    # scream if both missing. this is also where all the defults policies
    # take place. also handling the by_convs attribute carefully.
    scale_factors, out_shape, by_convs = set_scale_and_output_size(
        in_shape,
        out_shape,
        scale_factors,
        by_convs,
        scale_tolerance,
        max_numerator,
        eps,
        fw,
    )

    # sort indices of dimensions according to scale of each dimension.
    # since we are going dim by dim this is efficient
    sorted_filtered_dims_and_scales = [(dim, scale_factors[dim], by_convs[dim],
                                        in_shape[dim], out_shape[dim])
                                       for dim in sorted(range(n_dims),
                                       key=lambda ind: scale_factors[ind])
                                       if scale_factors[dim] != 1.]

    # unless support size is specified by the user, it is an attribute
    # of the interpolation mode
    if support_size is None:
        support_size = interp_mode.support_size

    # output begins identical to input and changes with each iteration
    output = input

    # iterate over dims
    for (dim, scale_factor, dim_by_convs, in_size, out_size
         ) in sorted_filtered_dims_and_scales:
        # STEP 1- PROJECTED GRID: The non-integer locations of the projection
        # of output pixel locations to the input tensor
        projected_grid = get_projected_grid(
            in_size,
            out_size,
            scale_factor,
            fw,
            dim_by_convs,
            device,
        )

        # STEP 1.5: ANTIALIASING- If antialiasing is taking place, we modify
        # the window size and the interpolation mode (see inside function)
        cur_interp_mode, cur_support_size = apply_antialiasing_if_needed(
            interp_mode,
            support_size,
            scale_factor,
            antialiasing,
        )

        # STEP 2- FIELDS OF VIEW: for each output pixels, map the input pixels
        # that influence it. Also calculate needed padding and update grid
        # accoedingly
        field_of_view = get_field_of_view(
            projected_grid,
            cur_support_size,
            fw,
            eps,
            device,
        )

        # STEP 2.5- CALCULATE PAD AND UPDATE: according to the field of view,
        # the input should be padded to handle the boundaries, coordinates
        # should be updated. actual padding only occurs when weights are
        # aplied (step 4). if using by_convs for this dim, then we need to
        # calc right and left boundaries for each filter instead.
        pad_size, projected_grid, field_of_view = calc_pad_size(
            in_size,
            out_size,
            field_of_view,
            projected_grid,
            scale_factor,
            dim_by_convs,
            fw,
            device,
        )

        # STEP 3- CALCULATE WEIGHTS: Match a set of weights to the pixels in
        # the field of view for each output pixel
        weights = get_weights(cur_interp_mode, projected_grid, field_of_view)

        # STEP 4- APPLY WEIGHTS: Each output pixel is calculated by multiplying
        # its set of weights with the pixel values in its field of view.
        # We now multiply the fields of view with their matching weights.
        # We do this by tensor multiplication and broadcasting.
        # if by_convs is true for this dim, then we do this action by
        # convolutions. this is equivalent but faster.
        if not dim_by_convs:
            output = apply_weights(
                output,
                field_of_view,
                weights,
                dim,
                n_dims,
                pad_size,
                pad_mode,
                fw,
            )
        else:
            output = apply_convs(
                output,
                scale_factor,
                in_size,
                out_size,
                weights,
                dim,
                pad_size,
                pad_mode,
                fw,
            )
    return output


def get_projected_grid(in_size, out_size, scale_factor, fw, by_convs, device=None):
    # we start by having the ouput coordinates which are just integer locations
    # in the special case when usin by_convs, we only need two cycles of grid
    # points. the first and last.
    grid_size = out_size if not by_convs else scale_factor.numerator
    out_coordinates = fw_arange(grid_size, fw, device)

    # This is projecting the ouput pixel locations in 1d to the input tensor,
    # as non-integer locations.
    # the following fomrula is derived in the paper
    # "From Discrete to Continuous Convolutions" by Shocher et al.
    return (out_coordinates / float(scale_factor)) + ((in_size - 1) / 2) - (
        (out_size - 1) / (2 * float(scale_factor))
    )


def get_field_of_view(projected_grid, cur_support_size, fw, eps, device):
    # for each output pixel, map which input pixels influence it, in 1d.
    # we start by calculating the leftmost neighbor, using half of the window
    # size (eps is for when boundary is exact int)
    left_boundaries = fw_ceil(projected_grid - cur_support_size / 2 - eps, fw)

    # then we simply take all the pixel centers in the field by counting
    # window size pixels from the left boundary
    ordinal_numbers = fw_arange(math.ceil(cur_support_size - eps), fw, device)
    return left_boundaries[:, None] + ordinal_numbers


def calc_pad_size(in_size, out_size, field_of_view, projected_grid, scale_factor,
                dim_by_convs, fw, device):
    if not dim_by_convs:
        # determine padding according to neighbor coords out of bound.
        # this is a generalized notion of padding, when pad<0 it means crop
        pad_size = [-field_of_view[0, 0].item(),
                  field_of_view[-1, -1].item() - in_size + 1]

        # since input image will be changed by padding, coordinates of both
        # field_of_view and projected_grid need to be updated
        field_of_view += pad_size[0]
        projected_grid += pad_size[0]

    else:
        # only used for by_convs, to calc the boundaries of each filter the
        # number of distinct convolutions is the numerator of the scale factor
        num_convs, stride = scale_factor.numerator, scale_factor.denominator

        # calculate left and right boundaries for each conv. left can also be
        # negative right can be bigger than in_size. such cases imply padding if
        # needed. however if# both are in-bounds, it means we need to crop,
        # practically apply the conv only on part of the image.
        left_pads = -field_of_view[:, 0]

        # next calc is tricky, explanation by rows:
        # 1) counting output pixels between the first position of each filter
        #    to the right boundary of the input
        # 2) dividing it by number of filters to count how many 'jumps'
        #    each filter does
        # 3) multiplying by the stride gives us the distance over the input
        #    coords done by all these jumps for each filter
        # 4) to this distance we add the right boundary of the filter when
        #    placed in its leftmost position. so now we get the right boundary
        #    of that filter in input coord.
        # 5) the padding size needed is obtained by subtracting the rightmost
        #    input coordinate. if the result is positive padding is needed. if
        #    negative then negative padding means shaving off pixel columns.
        right_pads = (((out_size - fw_arange(num_convs, fw, device) - 1)  # (1)
                      // num_convs)  # (2)
                      * stride  # (3)
                      + field_of_view[:, -1]  # (4)
                      - in_size + 1)  # (5)

        # in the by_convs case pad_size is a list of left-right pairs. one per
        # each filter

        pad_size = list(zip(left_pads, right_pads))
    return pad_size, projected_grid, field_of_view


def get_weights(interp_mode, projected_grid, field_of_view):
    # the set of weights per each output pixels is the result of the chosen
    # interpolation mode applied to the distances between projected grid
    # locations and the pixel-centers in the field of view (distances are
    # directed, can be positive or negative)
    weights = interp_mode(projected_grid[:, None] - field_of_view)

    # we now carefully normalize the weights to sum to 1 per each output pixel
    sum_weights = weights.sum(1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    return weights / sum_weights


def apply_weights(input, field_of_view, weights, dim, n_dims, pad_size, pad_mode,
                  fw):
    # for this operation we assume the resized dim is the first one.
    # so we transpose and will transpose back after multiplying
    tmp_input = fw_swapaxes(input, dim, 0, fw)

    # apply padding
    tmp_input = fw_pad(tmp_input, fw, pad_size, pad_mode)

    # field_of_view is a tensor of order 2: for each output (1d location
    # along cur dim)- a list of 1d neighbors locations.
    # note that this whole operations is applied to each dim separately,
    # this is why it is all in 1d.
    # neighbors = tmp_input[field_of_view] is a tensor of order image_dims+1:
    # for each output pixel (this time indicated in all dims), these are the
    # values of the neighbors in the 1d field of view. note that we only
    # consider neighbors along the current dim, but such set exists for every
    # multi-dim location, hence the final tensor order is image_dims+1.
    neighbors = tmp_input[field_of_view]

    # weights is an order 2 tensor: for each output location along 1d- a list
    # of weights matching the field of view. we augment it with ones, for
    # broadcasting, so that when multiplies some tensor the weights affect
    # only its first dim.
    tmp_weights = fw.reshape(weights, (*weights.shape, * [1] * (n_dims - 1)))

    # now we simply multiply the weights with the neighbors, and then sum
    # along the field of view, to get a single value per out pixel
    tmp_output = (neighbors * tmp_weights.to(neighbors.device)).sum(1)

    # we transpose back the resized dim to its original position
    return fw_swapaxes(tmp_output, 0, dim, fw)


def apply_convs(input, scale_factor, in_size, out_size, weights, dim, pad_size,
                pad_mode, fw):
    # for this operations we assume the resized dim is the last one.
    # so we transpose and will transpose back after multiplying
    input = fw_swapaxes(input, dim, -1, fw)

    # the stride for all convs is the denominator of the scale factor
    stride, num_convs = scale_factor.denominator, scale_factor.numerator

    # prepare an empty tensor for the output
    tmp_out_shape = list(input.shape)
    tmp_out_shape[-1] = out_size
    tmp_output = fw_empty(tuple(tmp_out_shape), fw, input.device)

    # iterate over the conv operations. we have as many as the numerator
    # of the scale-factor. for each we need boundaries and a filter.
    for conv_ind, (pad_size, filt) in enumerate(zip(pad_size, weights)):
        # apply padding (we pad last dim, padding can be negative)
        pad_dim = input.ndim - 1
        tmp_input = fw_pad(input, fw, pad_size, pad_mode, dim=pad_dim)

        # apply convolution over last dim. store in the output tensor with
        # positional strides so that when the loop is comlete conv results are
        # interwind
        tmp_output[..., conv_ind::num_convs] = fw_conv(tmp_input, filt, stride)

    return fw_swapaxes(tmp_output, -1, dim, fw)


def set_scale_and_output_size(in_shape, out_shape, scale_factors, by_convs,
                         scale_tolerance, max_numerator, eps, fw):
    # eventually we must have both scale-factors and out-sizes for all in/out
    # dims. however, we support many possible partial arguments
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be "
                         "provided")
    if out_shape is not None:
        # if out_shape has less dims than in_shape, we defaultly resize the
        # first dims for numpy and last dims for torch
        out_shape = (list(out_shape) + list(in_shape[len(out_shape):])
                     if fw is np
                     else list(in_shape[:-len(out_shape)]) + list(out_shape))
        if scale_factors is None:
            # if no scale given, we calculate it as the out to in ratio
            # (not recomended)
            scale_factors = [out_size / in_size for out_size, in_size
                             in zip(out_shape, in_shape)]
    if scale_factors is not None:
        # by default, if a single number is given as scale, we assume resizing
        # two dims (most common are images with 2 spatial dims)
        scale_factors = (scale_factors
                         if isinstance(scale_factors, (list, tuple))
                         else [scale_factors, scale_factors])
        # if less scale_factors than in_shape dims, we defaultly resize the
        # first dims for numpy and last dims for torch
        scale_factors = (list(scale_factors) + [1] *
                         (len(in_shape) - len(scale_factors)) if fw is np
                         else [1] * (len(in_shape) - len(scale_factors)) +
                         list(scale_factors))
        if out_shape is None:
            # when no out_shape given, it is calculated by multiplying the
            # scale by the in_shape (not recomended)
            out_shape = [math.ceil(scale_factor * in_size)
                         for scale_factor, in_size in
                         zip(scale_factors, in_shape)]
        # next part intentionally after out_shape determined for stability
        # we fix by_convs to be a list of truth values in case it is not
        if not isinstance(by_convs, (list, tuple)):
            by_convs = [by_convs] * len(out_shape)

        # next loop fixes the scale for each dim to be either frac or float.
        # this is determined by by_convs and by tolerance for scale accuracy.
        for ind, (sf, dim_by_convs) in enumerate(zip(scale_factors, by_convs)):
            # first we fractionaize
            if dim_by_convs:
                frac = Fraction(1/sf).limit_denominator(max_numerator)
                frac = Fraction(numerator=frac.denominator, denominator=frac.numerator)

            # if accuracy is within tolerance scale will be frac. if not, then
            # it will be float and the by_convs attr will be set false for
            # this dim
            if scale_tolerance is None:
                scale_tolerance = eps
            if dim_by_convs and abs(frac - sf) < scale_tolerance:
                scale_factors[ind] = frac
            else:
                scale_factors[ind] = float(sf)
                by_convs[ind] = False

        return scale_factors, out_shape, by_convs


def apply_antialiasing_if_needed(
        interp_mode, support_size, scale_factor, antialiasing,
    ):
    # antialiasing is "stretching" the field of view according to the scale
    # factor (only for downscaling). this is low-pass filtering. this
    # requires modifying both the interpolation (stretching the 1d
    # function and multiplying by the scale-factor) and the window size.
    scale_factor = float(scale_factor)
    if scale_factor >= 1.0 or not antialiasing:
        return interp_mode, support_size
    cur_interp_mode = (lambda arg: scale_factor *
                         interp_mode(scale_factor * arg))
    cur_support_size = support_size / scale_factor
    return cur_interp_mode, cur_support_size


def fw_ceil(x, fw):
    if fw is np:
        return fw.int_(fw.ceil(x))
    else:
        return x.ceil().long()


def fw_floor(x, fw):
    if fw is np:
        return fw.int_(fw.floor(x))
    else:
        return x.floor().long()


def fw_cat(x, fw):
    if fw is np:
        return fw.concatenate(x)
    else:
        return fw.cat(x)


def fw_swapaxes(x, ax_1, ax_2, fw):
    if fw is np:
        return fw.swapaxes(x, ax_1, ax_2)
    else:
        return x.transpose(ax_1, ax_2)


def fw_pad(x, fw, pad_size, pad_mode, dim=0):
    if pad_size == (0, 0):
        return x
    if fw is np:
        pad_vec = [(0, 0)] * x.ndim
        pad_vec[dim] = pad_size
        return fw.pad(x, pad_width=pad_vec, mode=pad_mode)
    else:
        if x.ndim < 3:
            x = x[None, None, ...]

        pad_vec = [0] * ((x.ndim - 2) * 2)
        pad_vec[0:2] = pad_size
        return fw.nn.functional.pad(x.transpose(dim, -1), pad=pad_vec,
                                    mode=pad_mode).transpose(dim, -1)


def fw_conv(input, filter, stride):
    # we want to apply 1d conv to any nd array. the way to do it is to reshape
    # the input to a 4D tensor. first two dims are singeletons, 3rd dim stores
    # all the spatial dims that we are not convolving along now. then we can
    # apply conv2d with a 1xK filter. This convolves the same way all the other
    # dims stored in the 3d dim. like depthwise conv over these.
    # TODO: numpy support
    reshaped_input = input.reshape(1, 1, -1, input.shape[-1])
    reshaped_output = torch.nn.functional.conv2d(reshaped_input,
                                                 filter.view(1, 1, 1, -1),
                                                 stride=(1, stride))
    return reshaped_output.reshape(*input.shape[:-1], -1)


def fw_arange(upper_bound, fw, device):
    if fw is np:
        return fw.arange(upper_bound)
    else:
        return fw.arange(upper_bound, device=device)


def fw_empty(shape, fw, device):
    if fw is np:
        return fw.empty(shape)
    else:
        return fw.empty(size=(*shape,), device=device)


def downsample_then_upsample(x, scale_factor, mode="area-bicubic"):
    ori_device = x.device
    if ori_device.type == "mps":
        temp_device = torch.device("cpu")

    if mode == "resizeright":
        x = resize(x, scale_factors=1 / scale_factor)
    elif mode == "area-bicubic":
        x = F.interpolate(x, scale_factor=1 / scale_factor, mode="area")
    else:
        if mode == "bicubic":
            x = x.to(temp_device)
        x = F.interpolate(x, scale_factor=1 / scale_factor, mode=mode)

    if mode == "resizeright":
        x = resize(x, scale_factors=scale_factor)
    elif mode == "area-bicubic":
        x = F.interpolate(
            x.to(temp_device), scale_factor=scale_factor, mode="bicubic",
        )
    else:
        x = F.interpolate(x, scale_factor=scale_factor, mode=mode).to(ori_device)
    return x.to(ori_device)


if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms.functional as TF
    from pathlib import Path

    # DEVICE = torch.device("cpu")
    DEVICE = get_device()
    # x = torch.randn(4, 3, 32, 32, device=DEVICE)
    # F.interpolate(x, 4, mode="bicubic")
    ROOT_DIR = Path(__file__).resolve().parent
    SAVE_DIR = ROOT_DIR/"experiments/image_resizing_modes"

    image = Image.open(SAVE_DIR/"ori.jpg").convert("RGB")
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    image = image.to(DEVICE)

    scale_factor = 64
    for mode in [
        "resizeright", "nearest", "bilinear", "bicubic", "area", "area-bicubic",
    ]:
        out = downsample_then_upsample(
            image[None, ...], scale_factor=scale_factor, mode=mode,
        )
        out_grid = image_to_grid(out, n_cols=1)
        save_image(out_grid, save_path=SAVE_DIR/f"{mode}.jpg")
