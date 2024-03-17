# 1. Samples
## 1) `"single_ref"` Mode
- The top left is each reference image.
- `dataset="celeba"`:
    - `scale_factor=4`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/d348a564-6632-4b0d-ae27-b881ee022844" width="450">
    - `scale_factor=8`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/80b76172-57da-4ac6-bb50-08567af8e903" width="450">
    - `scale_factor=16`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/1c282aa4-82df-461a-b7d6-5e26f86440b8" width="450">
    - `scale_factor=32`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/f3ceac00-f26d-495e-8bcd-540e8a3f4a33" width="450">
    - `scale_factor=64`:
        - <img src="https://github.com/KimRass/ILVR/assets/67457712/2e1bb2bd-388e-496d-bf52-cf640dcfbb2c" width="450">
## 2) `"various_scale_factors"` Mode
- The leftmost is each reference image and the rest correspond to `scale_factor=4`, `8`, `16`, `32`, `64` from left to right.
- `dataset="celeba"`:
    - <img src="https://github.com/KimRass/ILVR/assets/67457712/923b4b70-e0ef-4c79-9ff2-14a330d9a768" width="450">
    - <img src="https://github.com/KimRass/ILVR/assets/67457712/426bcd77-6186-49b6-aee5-4d87925174dd" width="450">
## 3) `"various_cond_range"` Mode
- The leftmost is each reference image and the rest correspond to ILVR on steps from 1000 to 0, to 125, 250, 375, 500, 625, 750, 875, 1000 (No ILVR steps) from left to right.
- `dataset="celeba"`:
    - <img src="https://github.com/KimRass/ILVR/assets/67457712/800a020f-0884-4bbf-b474-2ffe7ea2a672" width="700">
    - <img src="https://github.com/KimRass/ILVR/assets/67457712/8f87e077-1e7b-4cce-b2da-5caf37c484d6" width="700">
    - <img src="https://github.com/KimRass/ILVR/assets/67457712/0531a3c3-50e0-477b-a679-531b2a9943ef" width="700">

# 2. Implementation Details

# 3. Theoretical Background
$${x^{\prime}_{t - 1}} \sim p_{\theta}(x^{\prime}_{t - 1} \vert x_{t})$$
$$y_{t - 1} \sim q(y_{t - 1} \vert y)$$
$$x_{t - 1} \leftarrow \phi_{N}(y_{t - 1}) + x^{\prime}_{t - 1} - \phi_{N}(x^{\prime}_{t - 1})$$

# 4. References
- [1] https://github.com/assafshocher/ResizeRight
