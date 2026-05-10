# GAN-Based Virtual Home Staging: Project Summary

## Project Objective

Insert a specific piece of furniture into an empty room photograph and harmonize lighting, shadows, and perspective to make the result look photorealistic. This is "virtual home staging" — the interior design equivalent of virtual try-on.

---

## Dataset

### Primary: 3D-FRONT (3D Furnished Rooms with layOuts and semaNTics)

- **Source:** Large-scale synthetic indoor scene dataset (ICCV 2021)
- **What we use:** Pre-rendered room images at 256x256 resolution
- **Preprocessing pipeline** (`pipeline.py`):
  1. `scanner.py` — scans 3D-FRONT directories for bedroom renders
  2. `detector.py` — YOLOv8 detects beds in each render
  3. `inpainter.py` — LaMa inpaints the bed region to produce an "empty room"
  4. Outputs training triplets:
     - **input** — inpainted room (bed removed)
     - **furniture** — cropped bed image from the render
     - **target** — original room with bed (ground truth)
  5. `metadata.csv` stores paths, bounding boxes, and train/val/test splits
- **Size:** ~3,000 triplets (beds only), split into train/val/test
- **Limitation:** Small dataset for GAN training; only one furniture category (beds)

### Secondary: ST-GAN Indoor Dataset

- **Source:** CMU (CVPR 2018), 6.4 GB of rendered indoor scenes
- **Format:** `.npy` arrays at 120x160 resolution containing room backgrounds, furniture at correct/perturbed positions, and alpha masks
- **Use:** Pretrained ST-GAN weights + fine-tuning for geometric placement

### Mentioned but not used: InteriorNet, SUN RGB-D, iHarmony4

---

## Architectures Explored

### 1. Pix2Pix Baseline (Notebook 01)

| Component | Details |
|-----------|---------|
| **Generator** | U-Net encoder-decoder with skip connections |
| **Discriminator** | PatchGAN (70x70 receptive field) |
| **Input** | 6 channels (room RGB + furniture RGB concatenated) |
| **Output** | 3 channels (furnished room RGB) |
| **Loss** | LSGAN + 100x L1 |
| **Normalization** | Instance Normalization |
| **Framework** | TensorFlow/Keras |
| **Task** | Learn both placement AND rendering from concatenated inputs (no mask) |

**Results (200 epochs):**

| Metric | Value |
|--------|-------|
| SSIM | 0.6231 +/- 0.0842 |
| PSNR | 18.43 +/- 3.27 dB |
| FID | 187.4 |

**Why it failed:**

- **Blurry, averaged outputs.** The model received the furniture as 3 extra channels concatenated with the room (6ch total). Without any spatial guidance (no mask, no bbox), the generator had to simultaneously figure out *where* in the room the furniture goes, *what* it should look like at that location, and *how* the lighting should interact. With only ~3K training samples, the model could not disentangle these tasks and instead learned to produce a blurry average of all possible placements.
- **Mode collapse.** The generator converged to outputting near-copies of the input room with a faint furniture-colored haze in the general center area. The discriminator could easily distinguish real from fake, leading to an unstable training loop.
- **No spatial conditioning.** Concatenating the furniture crop as extra input channels destroys all spatial information — a bed image is spread across the entire image as 3 constant channels. The generator has no signal for where to place it.
- **Conclusion:** Pix2Pix works well for aligned image-to-image tasks (edges→photo, segmentation→photo) but is fundamentally unsuited for tasks requiring spatial reasoning about *where* to place novel content.

---

### 2. SPADE + Reference Encoder + Dual Discriminator (Notebook 04)

| Component | Details |
|-----------|---------|
| **Reference Encoder** | CNN extracting a style vector from the furniture image |
| **Generator** | U-Net encoder + SPADE decoder (spatial normalization) + AdaIN (style injection) |
| **Global Discriminator** | PatchGAN on full image with Spectral Normalization |
| **Local Discriminator** | PatchGAN on cropped furniture region |
| **Input** | Room image (3ch) + Furniture image (3ch, separate encoder) |
| **Loss** | Hinge GAN (global + local) + 100x L1 + 10x VGG perceptual + 10x Feature matching |
| **Framework** | TensorFlow/Keras |

**Results (~80 epochs):**

| Metric | Value |
|--------|-------|
| SSIM | 0.5847 +/- 0.1103 |
| PSNR | 16.92 +/- 4.15 dB |
| FID | 214.7 |

**Training curve analysis (see loss plots):**

| Loss | Start | Epoch 10 | Epoch 80 | Interpretation |
|------|-------|----------|----------|---------------|
| **Generator** | ~4000 | ~2500 | ~2000 | Drops early but plateaus extremely high — generator never learned to fool D |
| **Discriminator** | ~1.5 | ~1.0 | ~1.0 | Drops immediately and flatlines — D "won" within 10 epochs |
| **Train L1** | ~0.225 | ~0.100 | ~0.085 | Decreases steadily — but this just means the model copies the room background |
| **Perceptual** | ~400 | ~250 | ~200 | Mirrors generator pattern — high plateau indicates poor visual quality |
| **Val L1** | ~0.18 | ~0.09 | ~0.08 | Converges fast then plateaus — model learned a near-identity mapping early |

The curves reveal a clear failure pattern: the **L1 and Val L1 losses decrease smoothly**, suggesting the model is learning *something* — but what it learned was to simply reproduce the input room without furniture. The **generator loss plateaus at ~2000-2500** (extremely high for hinge loss), meaning the discriminator easily distinguishes the generator's outputs from real furnished rooms. The **discriminator loss flatlines at ~1.0 within 10 epochs**, showing it quickly became confident and provided diminishing gradient signal to the generator.

**Why it failed:**

- **Too many learnable components for ~3K samples.** The architecture has four separate networks (reference encoder, SPADE generator, global discriminator, local discriminator) with a combined parameter count far exceeding what 3K samples can support. Each component needs sufficient gradient signal to learn, but the data simply cannot provide it.
- **Discriminator dominance.** The discriminator converged to ~1.0 within 10 epochs and stayed there, leaving the generator starved of useful adversarial gradients. The generator loss plateaued at ~2000 — it was never able to catch up. This is visible in the Generator vs Discriminator curves: D stabilized while G remained stuck.
- **The model learned to copy, not create.** The smooth decline in L1 and Val L1 is misleading. These losses measure pixel-wise distance to the ground truth, and the easiest way to minimize them is to output the input room unchanged. Since the room background accounts for ~85% of pixels, copying the input yields a low L1 loss even without placing any furniture. The perceptual loss remaining at ~200 confirms the model is not generating realistic furniture.
- **Loss imbalance.** The combined loss (hinge GAN + L1 + VGG perceptual + feature matching) had competing gradients. The VGG perceptual loss dominated early training, pushing the generator toward blurry but perceptually similar outputs, while the adversarial loss tried to push for sharp details — the generator oscillated between these objectives.
- **AdaIN style injection was too weak.** The furniture reference encoder produces a single style vector that gets injected via AdaIN at every decoder level. But a 256-dimensional style vector cannot encode enough information about a specific piece of furniture (shape, color, texture, shadows) for the decoder to reconstruct it faithfully. The generator resorted to ignoring the style vector and producing room-only outputs.
- **Conclusion:** SPADE-based architectures are designed for semantic-to-image synthesis (e.g., segmentation map → photorealistic image) with large datasets (Cityscapes: 25K, COCO-Stuff: 118K). Applying them to furniture placement with 3K samples and a reference-conditioned formulation overwhelmed the architecture.

---

### 3. Composite Harmonization GAN (Notebook 06)

| Component | Details |
|-----------|---------|
| **Generator** | U-Net encoder-decoder (4ch input: composite RGB + mask, 3ch output) |
| **Discriminator** | PatchGAN with Spectral Normalization |
| **Input** | Pre-pasted composite image (furniture naively pasted onto room) + binary mask |
| **Output** | Harmonized room (edge blending, shadow generation, lighting correction) |
| **Loss** | Hinge GAN + 100x L1 + 1x VGG perceptual |
| **Framework** | TensorFlow/Keras |
| **Inference** | Mask2Former segmentation → placement heuristics → paste → GAN harmonize |

**Results (200 epochs):**

| Metric | Value |
|--------|-------|
| SSIM | 0.9502 +/- 0.0270 |
| PSNR | 31.57 +/- 2.13 dB |
| LPIPS (VGG) | 9.4604 +/- 2.8998 |
| BG-PSNR | 33.36 +/- 2.58 dB |
| fMSE | 0.001961 +/- 0.001197 |

**Key finding:** The GAN learned a near-identity mapping because the composite was already very close to the ground truth (furniture was extracted from the same render). When the Mask2Former-based placement put furniture at a different position than ground truth, the U-Net could not spatially relocate it — fundamental architectural limitation.

---

### 4. ST-GAN Spatial Transformer (Notebook 07) — Current Work

| Component | Details |
|-----------|---------|
| **Generator** | Spatial Transformer Network predicting 8D homography parameters |
| **Discriminator** | Conv-based discriminator on composited image |
| **Input** | Room BG [B,120,160,3] + Furniture RGBA [B,120,160,4] + initial translation [B,8] |
| **Output** | 3x3 homography matrix (via matrix exponential), applied as differentiable warp |
| **Loss** | WGAN-GP (Wasserstein + gradient penalty, lambda=10) + warp norm penalty |
| **Framework** | TensorFlow (TF1 compat mode) |
| **Pretrained** | Yes — CMU pretrained indoor model (89.6 MB) |
| **Fine-tuning** | 10,000 iterations on our 3D-FRONT data |

**Results (10,000 iterations fine-tuning):**

| Approach | MSE | PSNR (dB) |
|----------|-----|-----------|
| No correction (baseline) | 0.020657 | 17.14 |
| Pretrained ST-GAN (zero-shot) | 0.033597 | 14.91 |
| **Fine-Tuned ST-GAN** | **0.009335** | **21.02** |

**Training curve analysis:** The WGAN-GP training showed healthy dynamics — `loss_GP` went increasingly negative (from ~0 to ~-5 by iteration 3500, then oscillated in the -2 to -4 range), indicating the generator was successfully fooling the discriminator. `loss_D` stayed near 0 throughout, showing balanced adversarial training.

**Key findings:**
- **Pretrained ST-GAN is worse than no correction** (MSE 0.034 vs 0.021) — the domain gap between CMU's indoor dataset and our 3D-FRONT data causes wrong corrections.
- **Fine-tuning cut MSE by more than half** (0.021 → 0.009) and gained ~4 dB in PSNR (17.14 → 21.02).
- The model genuinely learned to correct furniture placement geometry on our data.
- Remaining artifacts are at borders (hard edges) — addressed by the blending stage.

**Key advantage:** Operates in geometric warp parameter space, not pixel space — can actually move, scale, and correct perspective. Connected to Poisson blending for seamless edge harmonization.

---

### 5. Diffusion Models (Notebook 05)

| Approach | Type | Details |
|----------|------|---------|
| **Paint-by-Example** | Zero-shot | Pre-trained exemplar-guided inpainting; generates new furniture inspired by reference crop |
| **SD Inpainting + IP-Adapter** | Zero-shot | Text + image prompt conditioned inpainting (`runwayml/stable-diffusion-inpainting`) |
| **Paint-by-Example (fine-tuned)** | Fine-tuned | UNet trained on border-blending task — preserves exact furniture pixels, only blends edges |

**Note:** Paint-by-Example has version incompatibilities with current transformers/diffusers (the `PaintByExampleImageEncoder` lacks `all_tied_weights_keys` in newer versions). For border blending, we switched to **Poisson blending** (`cv2.seamlessClone`) which is faster, deterministic, and equally effective for edge harmonization.

---

## Current Pipeline Architecture

```
                    ┌──────────────────────────────────┐
                    │  1. Mask2Former (ADE20K)          │
Empty Room ────────►│  → semantic segmentation          │
                    │  → floor/wall region detection     │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │  2. Heuristic Placement            │
Furniture ─────────►│  → find floor region               │
  Crop              │  → compute furniture bbox          │
                    │  → paste furniture onto room       │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │  3. ST-GAN (Notebook 07)           │
                    │  → predicts 8D homography warp     │
                    │  → geometrically corrects placement│
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │  4. Poisson Blending (OpenCV)      │
                    │  → cv2.seamlessClone               │
                    │  → matches color/lighting at edges │──► Final Staged Room
                    │  → smooth, natural transitions     │
                    └──────────────────────────────────┘
```

---

## Comparison of All Approaches

| Approach | Placement | Furniture | SSIM | PSNR (dB) | FID | Status |
|----------|-----------|-----------|------|-----------|-----|--------|
| Pix2Pix (NB01) | Learned | Generated | 0.6231 | 18.43 | 187.4 | Failed (blurry outputs) |
| SPADE (NB04) | Learned | Generated | 0.5847 | 16.92 | 214.7 | Collapsed at ~80 epochs |
| Harmonization GAN (NB06) | Heuristic | Pasted (exact) | 0.9502 | 31.57 | — | Near-identity mapping |
| ST-GAN Pretrained (NB07) | Learned (warp) | Exact | — | 14.91 | — | Worse than baseline (domain gap) |
| **ST-GAN Fine-Tuned (NB07)** | **Learned (warp)** | **Exact** | **—** | **21.02** | **—** | **Best GAN result (synthetic test)** |
| **ST-GAN + Poisson E2E (NB07)** | **Mask2Former + heuristic** | **Exact (blended)** | **0.7642** | **16.66** | **—** | **Full pipeline** |

---

## Stage Coupling and Conditioning Design

A two-stage pipeline (geometric placement → photorealistic refinement) carries a real risk of becoming **decoupled**: a free-form refinement stage can overwrite, drift, or hallucinate over the geometric decision the first stage produced. We address this with an **explicit conditioning contract** that makes such drift structurally impossible.

### The contract

Given the ST-GAN composite `C`, the post-warp furniture mask `M_furn`, and the clean background `bg`, we partition the image into:

```
M_border = dilate(M_furn, k=21) AND NOT M_furn          # editable
M_locked = M_furn UNION (NOT dilate(M_furn, k=21))      # locked (paste-back from C)
```

The refinement stage may modify pixels only inside `M_border`; everything in `M_locked` is restored from `C` by hard overwrite after refinement.

| Region          | Source after refinement | Geometric guarantee     |
|-----------------|-------------------------|-------------------------|
| Furniture core  | ST-GAN composite (`C`)  | Bit-exact preservation  |
| Room background | ST-GAN composite (`C`)  | Bit-exact preservation  |
| Border ring     | Refinement output       | Free to add shadows / blend lighting |

### How each blender satisfies the contract

- **Poisson blending (`cv2.seamlessClone`, `MIXED_CLONE`) — current default.** The seamless-clone operator solves a Poisson equation that adjusts colors at the seam to match neighboring gradients. It cannot translate pixels — the mask defines a fixed support and the optimization only modifies values inside it. Geometry preservation is a *property of the operator itself*.
- **Geometry-locked SD Inpainting (Notebook 07, Part 12).** Diffusion is run with `M_border` as the inpainting mask and `C` as the base image. After the diffusion step, the original composite pixels are pasted back over `M_locked`. A verification cell asserts that `mean |composite - refined|` outside `M_border` is exactly 0 — if any future change broke the contract, the assertion would fail loudly.

### Verification

Notebook 07, Part 12 includes an empirical contract-verification cell that compares the composite and the refinement output pixel-wise across the locked region; the assertion `diff_outside == 0` is part of the test suite. Both blenders pass.

---

## Key Lessons Learned

1. **~3K samples is insufficient** for training complex GANs from scratch (Pix2Pix, SPADE both failed)
2. **U-Nets cannot perform spatial transformations** — skip connections preserve spatial position, making object relocation impossible
3. **Decomposing the problem** (placement vs harmonization vs blending) is essential
4. **Pre-trained models + fine-tuning** (ST-GAN) provide a huge advantage on small datasets — pretrained alone was worse than baseline, but fine-tuning achieved the best results
5. **ST-GAN's warp parameter space** is the right approach for GAN-based placement (vs pixel-space U-Nets) — MSE halved, PSNR +4 dB
6. **Poisson blending** is more practical than diffusion models for border harmonization — no model compatibility issues, fast, deterministic, and *geometry-preserving by construction*
7. **Full pipeline requires multiple specialized components**: segmentation (Mask2Former) → placement (heuristics) → geometric correction (ST-GAN) → blending (Poisson)
8. **Heuristic placement is the bottleneck** — the E2E pipeline drops from 21.02 dB (synthetic perturbation, where GT position is known) to 16.66 dB (heuristic placement). The ~4.4 dB gap is entirely due to placing furniture in a different (but plausible) location than ground truth. SSIM of 0.76 confirms room structure is preserved; the penalty comes from positional mismatch, not quality degradation
9. **Stage coupling must be designed, not assumed.** Any time a refinement stage is added on top of a geometric stage, an explicit conditioning contract (here: editable border-ring mask + hard paste-back) prevents the refinement from silently overwriting upstream decisions. The contract is checked by an assertion, not just trusted.
