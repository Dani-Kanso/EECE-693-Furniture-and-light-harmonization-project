# Furniture Placement GAN -- Experimental Design Document

## 1. Problem Definition

**Task:** Given an empty room image and a reference furniture image (bed), generate a realistic room image with the furniture placed correctly.

**Formal definition:**

- x = empty room image (inpainted, bed removed)
- r = reference bed image (cropped from rerender)
- m = placement mask (binary, reconstructed from bbox)
- y = target image (original room with bed)

**Two problem variants:**

| Variant | Formulation | Description |
|---------|-------------|-------------|
| A (pixel generation) | G(x, m) -> y | Model learns a generic bed prior; cannot control which bed appears |
| B (conditioned generation) | G(x, r, m) -> y | Model learns to place THIS specific bed |
| C (geometric placement) | G(x, r) -> warp params | Model predicts how to geometrically transform the real furniture image for compositing |

---

## 2. Datasets

### 2.1 Primary: 3D-FRONT Processed Triplets

Source: Preprocessing pipeline (`pipeline.py`)

Per sample:
- `input` -- inpainted room (bed removed via LaMa)
- `target` -- original rerender (ground truth with bed)
- `furniture` -- cropped bed image from rerender (YOLOv8 bbox)
- `bbox` -- (x1, y1, x2, y2) bounding box coordinates
- `confidence` -- YOLOv8 detection confidence

Metadata stored in `metadata.csv`. Split by room (not by image) to prevent data leakage. Estimated ~3k training samples.

### 2.2 Fine-tuning: SUN RGB-D

10,335 real indoor RGB-D images with dense annotations (146,617 2D polygons, 64,595 3D bounding boxes). Used to fine-tune models on photorealistic data after initial training on 3D-FRONT synthetic renders. Bridges the synthetic-to-real domain gap.

### 2.3 Evaluation: OPA (Object Placement Assessment)

62,074 training + 11,396 test composite images with binary rationality labels. Used as an external benchmark to evaluate placement realism (is the object placement reasonable in terms of size, position, perspective, semantics?).

---

## 3. Experiments

### Experiment 1: Pix2Pix Baseline

**Notebook:** `01_pix2pix.ipynb`
**Variant:** A -- G(x, m) -> y
**Purpose:** Pipeline validation, establishes lower-bound baseline.

**Generator:** U-Net encoder-decoder with skip connections.
- 8 downsampling blocks: Conv2d(4x4, stride 2) -> InstanceNorm -> LeakyReLU(0.2)
- 8 upsampling blocks: ConvTranspose2d(4x4, stride 2) -> InstanceNorm -> ReLU + skip concat
- Dropout(0.5) on first 4 decoder blocks
- Final: ConvTranspose2d -> Tanh

**Discriminator:** PatchGAN with 70x70 receptive field.
- C64 -> C128 -> C256 -> C512 (stride 1) -> C1 (stride 1)
- InstanceNorm on all except first layer, LeakyReLU(0.2)
- Output: 30x30 patch map (for 256x256 input)

**Input:** concat(room_RGB, mask) = 4 channels
**Discriminator input:** concat(condition, real_or_fake) = 7 channels
**Output:** 3-channel RGB

**Loss:**
```
L_G = L_LSGAN + 100 * L_L1
L_D = 0.5 * (L_LSGAN_real + L_LSGAN_fake)
```

**Limitation:** Cannot control which bed appears. Learns a generic bed prior.

---

### Experiment 2: Pix2Pix + Reference Encoder

**Notebook:** `02_pix2pix_ref.ipynb`
**Variant:** B -- G(x, r, m) -> y
**Purpose:** First model that conditions on specific furniture appearance.

**Generator:** U-Net (same as Exp 1) with FiLM conditioning at decoder layers.

**Reference Encoder:** ResNet-18 (ImageNet pretrained, fine-tuned) processes the furniture crop.
- Output: 512-dimensional style vector
- Injection via FiLM (Feature-wise Linear Modulation):
  ```
  gamma = fc_gamma(style_vector)  # per-channel scale
  beta  = fc_beta(style_vector)   # per-channel shift
  features_out = gamma * features + beta
  ```

**Discriminator:** PatchGAN (same as Exp 1), input is 7 channels.

**Loss:**
```
L_G = L_LSGAN + 100 * L_L1 + 10 * L_perceptual
L_perceptual = sum of L1 distances at VGG-19 layers relu1_1, relu2_1, relu3_1, relu4_1
```

---

### Experiment 3: ST-GAN (Spatial Transformer GAN)

**Notebook:** `03_stgan.ipynb`
**Variant:** C -- geometric placement
**Purpose:** Fundamentally different approach: predicts geometric warp parameters instead of generating pixels. Preserves exact furniture appearance.

**Reference:** Lin et al., "ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing", CVPR 2018.

**Generator:** Iterative chain of N=4 Spatial Transformer Networks.
- Each STN predicts 8-dimensional homography warp parameters
- Architecture per STN: C(32)-C(64)-C(128)-C(256)-C(512)-L(256)-L(8)
- Input: 7 channels (foreground RGBA + background RGB)
- Output: 8 warp parameters (homography in sl(3) Lie algebra)
- Sequential training: train G1, freeze, train G2, freeze, ..., then fine-tune end-to-end

**Discriminator:** PatchGAN on composite image (3 channels).
- C(32)-C(64)-C(128)-C(256)-C(512)-C(1)
- No normalization layers

**Compositing:**
```
warped_furniture = homography_warp(furniture, params)
composite = warped_furniture * mask + background * (1 - mask)
```

**Loss:**
```
L_D = WGAN-GP objective
L_G = -D(composite) + lambda_update * ||delta_p||^2
```
The warp regularization term prevents trivial solutions (shrinking/translating furniture out of frame).

**Key advantages:**
- Resolution-independent (warp params transfer to any resolution)
- Exact furniture appearance preservation
- Learns semantic placement rules (beds against walls, etc.)

**Key limitations:**
- Homography only (no complex 3D deformations)
- No lighting/shadow integration
- No appearance harmonization

---

### Experiment 4: SPADE + Reference Encoder + Dual Discriminator

**Notebook:** `04_spade_ref.ipynb`
**Variant:** B -- G(x, r, m) -> y
**Purpose:** Best pixel-generation architecture. Strongest spatial control + appearance conditioning.

**Generator:** SPADE ResNet blocks.
- Each block contains SPADE normalization conditioned on the mask
- SPADE: learns per-pixel affine parameters from the mask at each layer
  ```
  gamma, beta = conv(mask_downsampled_to_feature_size)
  out = gamma * instance_norm(features) + beta
  ```
- Reference encoder (ResNet-18) injects furniture style via AdaIN into each SPADE block
- Room encoder: lightweight CNN that provides initial structure features

**Discriminator (dual):**
- Global: Multi-scale PatchGAN (2 scales: 1x and 0.5x) on full image
- Local: PatchGAN on cropped bed region (using bbox to extract patches from both real and fake)

**Loss:**
```
L_G = L_hinge_global + L_hinge_local + 10 * L_perceptual + 10 * L_feature_matching + 100 * L_L1
L_feature_matching = L1 distance between real and fake intermediate discriminator features
```

---

### Experiment 5: ST-GAN + Appearance Refinement Network

**Notebook:** `05_stgan_refine.ipynb`
**Variant:** C + refinement
**Purpose:** Combines geometric accuracy of ST-GAN with appearance quality of pixel generation.

**Stage 1:** ST-GAN (same as Exp 3) produces a geometrically correct composite.

**Stage 2:** Refinement U-Net takes the composite and outputs a harmonized version.
- Input: ST-GAN composite (3ch) + mask (1ch) = 4 channels
- Output: refined composite (3ch)
- Adjusts lighting, shadows, blending at boundaries
- Trained with the ST-GAN frozen (only refinement network trains)

**Discriminator:** PatchGAN on refined composite vs. real images.

**Loss (Stage 2 only):**
```
L_refine = L_LSGAN + 100 * L_L1 + 10 * L_perceptual
```

**Training:** Two-phase. First train ST-GAN to convergence. Then freeze ST-GAN and train refinement network.

---

### Experiment 6: Local Insertion GAN

**Notebook:** `06_local_insertion.ipynb`
**Variant:** B (local) -- generates only the furniture region
**Purpose:** Easiest learning task. Background is pixel-perfect by construction.

**Generator:** Encoder-decoder that generates only the bed patch.
- Input: local room context around bbox + furniture reference
- Output: generated bed patch (same size as bbox region)

**Compositing:**
```
alpha = learned_blending_network(patch_boundary)
output = room * (1 - mask * alpha) + generated_patch * (mask * alpha)
```
Learned alpha blending at boundaries prevents hard seams.

**Discriminator:** Local PatchGAN on the insertion region only.

**Loss:**
```
L_G = L_LSGAN + 100 * L_L1 + 10 * L_perceptual  (all computed only on masked region)
```

**Advantages:** Smaller output space, less data needed, pixel-perfect background.
**Disadvantages:** Boundary artifacts, no global lighting adjustment, no shadows beyond mask.

---

### Experiment 7: Diffusion-GAN Hybrid (DDGAN-style)

**Notebook:** `07_ddgan.ipynb`
**Variant:** B -- conditioned generation
**Purpose:** Experimental. Tests whether a diffusion-GAN hybrid improves mode coverage and quality over pure GANs.

**Reference:** Xiao et al., "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs", ICLR 2022.

**Generator:** 2-4 step denoising process where each denoising step is a conditional GAN.
- Each step: takes noisy image + condition (room, mask, furniture ref) -> less noisy image
- Conditioning injected via cross-attention and concatenation
- Noise schedule: linear, T=4 steps

**Discriminator:** Multi-scale PatchGAN. Evaluates denoised output at each step.

**Loss:**
```
L_G = L_adversarial + L_denoising
L_D = WGAN-GP or hinge loss
```

**Risk:** High implementation complexity. No Keras/TF reference. With ~3k samples, added capacity may cause overfitting.

---

## 4. Harmonization Pipeline (Separate)

Post-processing applied AFTER any placement model to fix lighting, shadows, and color consistency.

### Option A: Fine-tuned U-Net Harmonizer

- Simple U-Net that takes composite + mask -> harmonized output
- Trained on SUN RGB-D: create synthetic composites (cut-paste with known ground truth)
- Fast inference, easy to train

### Option B: SIDNet (Shading-Aware)

- Decomposes harmonization into illumination estimation + foreground re-rendering
- Uses a shading-aware illumination descriptor
- Better physics-grounded approach

### Option C: LumiNet (Diffusion-Based, CVPR 2025)

- Modified ControlNet with intrinsic/extrinsic lighting representations
- Handles specular highlights, indirect illumination, cast shadows
- State-of-the-art but heaviest compute

### Fine-tuning Protocol

1. Pre-train harmonizer on synthetic composites from 3D-FRONT data
2. Fine-tune on SUN RGB-D real indoor images (cut-paste composites with GT)
3. Evaluate with and without harmonization for each placement model

---

## 5. Evaluation Metrics

### Image Quality

| Metric | What it measures | Direction | Scope |
|--------|-----------------|-----------|-------|
| FID | Distribution-level quality | Lower = better | Full test set |
| Local FID | Furniture region quality | Lower = better | Cropped bed regions |
| LPIPS | Perceptual similarity to GT | Lower = better | Per-image |
| SSIM | Structural similarity to GT | Higher = better | Per-image |
| PSNR | Pixel-level reconstruction | Higher = better | Per-image |

### Placement Quality

| Metric | What it measures | Direction | Scope |
|--------|-----------------|-----------|-------|
| OPA Score | Placement rationality | Higher = better | Per-image |
| SCSSIM | Scene composition structure | Higher = better | Per-image |
| Mask IoU | Placement location accuracy (ST-GAN) | Higher = better | Per-image |

### Background Preservation

| Metric | What it measures | Direction | Scope |
|--------|-----------------|-----------|-------|
| BG-PSNR | Background pixel preservation | Higher = better | Non-furniture region |
| BG-SSIM | Background structure preservation | Higher = better | Non-furniture region |

### Harmonization Quality

| Metric | What it measures | Direction | Scope |
|--------|-----------------|-----------|-------|
| fMSE | Foreground MSE after harmonization | Lower = better | Furniture region |

### Human Evaluation

- Visual Turing Test: show real vs. generated images to participants, measure fooling rate
- Preference study: show outputs from different models side-by-side, collect rankings

---

## 6. Benchmark Comparison Table

All experiments evaluated on the same held-out test split.

| Model | FID | Local FID | LPIPS | SSIM | PSNR | BG-PSNR | OPA |
|-------|-----|-----------|-------|------|------|---------|-----|
| Pix2Pix | - | - | - | - | - | - | - |
| Pix2Pix + Ref | - | - | - | - | - | - | - |
| ST-GAN | - | - | - | - | - | - | - |
| SPADE + Ref + Dual | - | - | - | - | - | - | - |
| ST-GAN + Refine | - | - | - | - | - | - | - |
| Local Insertion | - | - | - | - | - | - | - |
| DDGAN Hybrid | - | - | - | - | - | - | - |

### Additional comparisons:
- Each model with vs. without harmonization post-processing
- Each model with vs. without SUN RGB-D fine-tuning
- Ablation: effect of local discriminator (Exp 4)
- Ablation: effect of reference encoder (Exp 2 vs Exp 1)
- Ablation: number of ST-GAN warp stages (1, 2, 4)
- Ablation: FiLM vs. concatenation for reference injection (Exp 2)

---

## 7. Shared Training Protocol

| Parameter | Value |
|-----------|-------|
| Resolution | 256 x 256 |
| Batch size | 8-16 (GPU dependent) |
| Optimizer | Adam |
| Learning rate | 2e-4 (G and D) |
| Adam betas | (0.5, 0.999) |
| LR schedule | Linear decay over last 50% of training |
| Epochs | 200 (Pix2Pix/SPADE), 300 (ST-GAN) |
| Normalization (G) | InstanceNorm (except ST-GAN: none) |
| Normalization (D) | SpectralNorm or InstanceNorm |
| Weight init | N(0, 0.02) for conv layers |
| Image normalization | [-1, 1] (Tanh output) |
| Data augmentation | Random horizontal flip |

---

## 8. Build Order

| Step | Deliverable | Depends on |
|------|-------------|------------|
| 1 | `EXPERIMENTS.md` (this document) | -- |
| 2 | `01_pix2pix.ipynb` | Data from pipeline |
| 3 | `02_pix2pix_ref.ipynb` | Exp 1 validated |
| 4 | `03_stgan.ipynb` | Exp 1 validated |
| 5 | `04_spade_ref.ipynb` | Exp 2 concepts |
| 6 | `05_stgan_refine.ipynb` | Exp 3 trained |
| 7 | `06_local_insertion.ipynb` | Exp 1 validated |
| 8 | `07_ddgan.ipynb` | All others done |
| 9 | Harmonization pipeline | Best placement model chosen |
| 10 | Final comparison + paper figures | All experiments |

---

## 9. Architectures NOT Used (and Why)

| Model | Reason for exclusion |
|-------|---------------------|
| Vanilla GAN | Unconditional -- cannot accept room/mask input |
| CycleGAN | We have paired data; cycle consistency adds unnecessary complexity |
| StyleGAN/StyleGAN2 | Unconditional generation; not designed for spatial control |
| OASIS | Requires dense semantic segmentation maps; we only have bounding boxes |
| STGAN (Selective Transfer) | Designed for facial attribute editing; wrong control signal for spatial insertion |
| Pix2PixHD | Only beneficial at 512+ resolution; revisit after 256x256 models converge |
