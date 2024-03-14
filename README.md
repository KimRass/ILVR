# 1. Samples
## 1) `"single_ref"` Mode
- `dataset="celeba"`:
    - `scale_factor=4`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/d348a564-6632-4b0d-ae27-b881ee022844" width="700">
    - `scale_factor=8`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/80b76172-57da-4ac6-bb50-08567af8e903" width="700">
    - `scale_factor=16`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/1c282aa4-82df-461a-b7d6-5e26f86440b8" width="700">
    - `scale_factor=8`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/f3ceac00-f26d-495e-8bcd-540e8a3f4a33" width="700">
## 2) `"various_scale_factors"` Mode
- <img src="https://github.com/KimRass/ILVR/assets/67457712/7cbb993a-9764-4b7c-a406-eac1ed5e1250" width="500">

<!-- - <img src="" width="700">
- <img src="" width="700">
- <img src="" width="700">
- <img src="" width="700">
- <img src="" width="700">
- <img src="" width="700"> -->

# 2. Theoretical Background
$${x^{\prime}_{t - 1}} \sim p_{\theta}(x^{\prime}_{t - 1} \vert x_{t})$$
$$y_{t - 1} \sim q(y_{t - 1} \vert y)$$
$$x_{t - 1} \leftarrow \phi_{N}(y_{t - 1}) + x^{\prime}_{t - 1} - \phi_{N}(x^{\prime}_{t - 1})$$
