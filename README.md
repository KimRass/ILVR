# 1. Samples
## 1) `"single_ref"` Mode
- The top left is each reference image.
- `dataset="celeba"`:
    | <img src="https://github.com/KimRass/ILVR/assets/67457712/d348a564-6632-4b0d-ae27-b881ee022844" width="450"> |
    |:-:|
    | `scale_factor=4` |

    | <img src="https://github.com/KimRass/ILVR/assets/67457712/80b76172-57da-4ac6-bb50-08567af8e903" width="450"> |
    |:-:|
    | `scale_factor=8` |

    | <img src="https://github.com/KimRass/ILVR/assets/67457712/1c282aa4-82df-461a-b7d6-5e26f86440b8" width="450"> |
    |:-:|
    | `scale_factor=16` |

    | <img src="https://github.com/KimRass/ILVR/assets/67457712/f3ceac00-f26d-495e-8bcd-540e8a3f4a33" width="450"> |
    |:-:|
    | `scale_factor=32` |

    | <img src="https://github.com/KimRass/ILVR/assets/67457712/2e1bb2bd-388e-496d-bf52-cf640dcfbb2c" width="450"> |
    |:-:|
    | `scale_factor=64` |
## 2) `"various_scale_factors"` Mode
- `dataset="celeba"`:

| <img src="https://github.com/KimRass/ILVR/assets/67457712/923b4b70-e0ef-4c79-9ff2-14a330d9a768" width="450"> |
|:-:|
| <img src="https://github.com/KimRass/ILVR/assets/67457712/426bcd77-6186-49b6-aee5-4d87925174dd" width="450"> |
| The leftmost is each reference image and the rest correspond to `scale_factor=4`, `8`, `16`, `32`, `64` from left to right. |
## 3) `"various_cond_range"` Mode
`dataset="celeba"`:
| <img src="https://github.com/KimRass/ILVR/assets/67457712/800a020f-0884-4bbf-b474-2ffe7ea2a672" width="700"> |
|:-:|
| <img src="https://github.com/KimRass/ILVR/assets/67457712/8f87e077-1e7b-4cce-b2da-5caf37c484d6" width="700"> |
| <img src="https://github.com/KimRass/ILVR/assets/67457712/0531a3c3-50e0-477b-a679-531b2a9943ef" width="700"> |
| The leftmost is each reference image and the rest correspond to ILVR on steps from 1000 to 0, to 125, 250, 375, 500, 625, 750, 875, 1000 (No ILVR steps) from left to right. |

# 2. Implementation Details

# 3. Experiments
## 1) Image Resizing Modes
| <img src="https://github.com/KimRass/ILVR/assets/67457712/3ce2b854-373d-4f56-a583-cc76971324fd" width="200"> |
|:-:|
| Original image |

| <img src="https://github.com/KimRass/ILVR/assets/67457712/601bf30c-5485-4041-b38b-afa79f6e5bdf" width="200"> | <img src="https://github.com/KimRass/ILVR/assets/67457712/3c6edb83-e187-4e34-8218-b88b19c66ef9" width="200"> | <img src="https://github.com/KimRass/ILVR/assets/67457712/54e8c2cb-c179-4a09-929c-a070914e3e8a" width="200"> | <img src="https://github.com/KimRass/ILVR/assets/67457712/55304975-0d20-4280-ae97-1c2e4ef13b33" width="200"> |
|:-:|-|:-:|-|
| Nearest | Area | Bilinear | Bicubic |
- Upsample과 Downsample 시에 동일한 Mode를 사용하면 Artifact가 발생합니다.

| <img src="https://github.com/KimRass/ILVR/assets/67457712/66f5d8a2-500b-42ec-9202-5f753122153e" width="200"> | <img src="https://github.com/KimRass/ILVR/assets/67457712/90586639-854c-4781-bb51-c2d55ae04313" width="200"> |
|:-:|-|
| Resizeright [1] | Area-Bicubic |
- Resizeright [1]를 사용할 경우 이미지에 자연스럽게 Blur를 적용한 듯한 효과가 생깁니다.
- Downsample시에 Area mode를, Upsample시에 Bicubic mode를 사용하면 이와 거의 유사한 효과를 낼 수 있습니다. 이쪽이 더 간단하게 구현되므로 본 구현체에서는 이 방법을 사용했습니다.

# 4 Theoretical Background
$${x^{\prime}_{t - 1}} \sim p_{\theta}(x^{\prime}_{t - 1} \vert x_{t})$$
$$y_{t - 1} \sim q(y_{t - 1} \vert y)$$
$$x_{t - 1} \leftarrow \phi_{N}(y_{t - 1}) + x^{\prime}_{t - 1} - \phi_{N}(x^{\prime}_{t - 1})$$

# 5. References
- [1] https://github.com/assafshocher/ResizeRight
