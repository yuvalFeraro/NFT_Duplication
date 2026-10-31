"""
generate_duplications.py
=========================
Production pipeline for generating synthetic NFT duplicates across 4 categories:
1. Geometric Transformations (Crop, Rotate, Flip, Aspect Ratio)
2. Color & Photometric Modifications (Brightness/Contrast, Hue/Saturation, Grayscale, Color Tint)
3. Noise & Compression Attacks (Gaussian Noise, Salt & Pepper, JPEG Artifacts, Blur)
4. Content Modifications (Watermark / Text Overlay, Borders, Background Modifications)

Follows all rules outlined in PROJECT_RULES.md.
"""

import os
import io
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont


# ==========================================
# Paths & Configuration
# ==========================================
BASE_DIR = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
COLLECTIONS_DIRS = {
    "Azuki": BASE_DIR / "Data_sets" / "Images_Azuki",
    "Bored_Ape_Yacht_Club": BASE_DIR / "Data_sets" / "Images_Bored_Ape_Yacht_Club",
    "Lil_Pudgys": BASE_DIR / "Data_sets" / "Images_Lil_Pudgys"
}

OUTPUT_BASE_DIR = BASE_DIR / "dataset_duplications"
CATEGORY_DIRS = {
    "Geometric": OUTPUT_BASE_DIR / "1_geometric",
    "Color": OUTPUT_BASE_DIR / "2_color",
    "Noise_Compression": OUTPUT_BASE_DIR / "3_noise_compression",
    "Content": OUTPUT_BASE_DIR / "4_content_modifications"
}

TARGET_COUNT_PER_CATEGORY = 500
RANDOM_SEED = 42


# ==========================================
# Category 1: Geometric Transformations
# ==========================================

# input = original image (PIL.Image)
# output = transformed image (PIL.Image), transformation name (str), transformation parameters (dict)

def apply_geometric(img: Image.Image) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Applies randomized geometric transformation."""
    subtype = random.choice(["Cropping", "Rotation", "Flipping", "AspectRatio"])
    w, h = img.size
    params: Dict[str, Any] = {"subtype": subtype}

    if subtype == "Cropping":
        # Random crop 5% to 30% from borders
        crop_left = int(w * random.uniform(0.05, 0.25))
        crop_top = int(h * random.uniform(0.05, 0.25))
        crop_right = int(w * (1.0 - random.uniform(0.05, 0.25)))
        crop_bottom = int(h * (1.0 - random.uniform(0.05, 0.25)))
        
        # Ensure valid box
        if crop_right <= crop_left + 20:
            crop_right = w - 10
            crop_left = 10
        if crop_bottom <= crop_top + 20:
            crop_bottom = h - 10
            crop_top = 10

        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        # Resize back to original dimensions
        result = cropped.resize((w, h), Image.Resampling.BICUBIC)
        params.update({"box": [crop_left, crop_top, crop_right, crop_bottom]})
        
        # for example
        #    params = {
        #     "subtype": "Cropping",
        #     "box": [15, 22, 210, 185]
        #       }


    elif subtype == "Rotation":
        # Either subtle continuous rotation (-15 to +15 deg) or cardinal (90, 180, 270)
        if random.random() < 0.20:
            angle = random.choice([90, 180, 270])
        else:
            angle = round(random.uniform(-15.0, 15.0), 2)
            if abs(angle) < 1.0:
                angle = 3.0 if angle >= 0 else -3.0

        # Rotate with expand and resize back
        rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        result = rotated.resize((w, h), Image.Resampling.BICUBIC)
        params.update({"angle_deg": angle})

    elif subtype == "Flipping":
        flip_type = "Horizontal" if random.random() < 0.85 else "Vertical"
        if flip_type == "Horizontal":
            result = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            result = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        params.update({"flip_axis": flip_type})

    else:  # AspectRatio
        scale_x = round(random.uniform(0.80, 1.25), 2)
        scale_y = round(random.uniform(0.80, 1.25), 2)
        new_w = max(50, int(w * scale_x))
        new_h = max(50, int(h * scale_y))
        distorted = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        result = distorted.resize((w, h), Image.Resampling.BICUBIC)
        params.update({"scale_x": scale_x, "scale_y": scale_y})

    return result, subtype, params


# ==========================================
# Category 2: Color & Photometric Modifications
# ==========================================
def apply_color(img: Image.Image) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Applies randomized color and photometric adjustment."""
    subtype = random.choice(["Brightness_Contrast", "Hue_Saturation", "Grayscale", "Color_Tint"])
    params: Dict[str, Any] = {"subtype": subtype}

    if subtype == "Brightness_Contrast":
        factor_b = round(random.uniform(0.60, 1.45), 2)
        factor_c = round(random.uniform(0.60, 1.45), 2)
        enhancer_b = ImageEnhance.Brightness(img)
        img_mod = enhancer_b.enhance(factor_b)
        enhancer_c = ImageEnhance.Contrast(img_mod)
        result = enhancer_c.enhance(factor_c)
        params.update({"brightness_factor": factor_b, "contrast_factor": factor_c})

    elif subtype == "Hue_Saturation":
        # Convert to HSV via numpy
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8)
        # Shift hue using PIL or numpy
        enhancer_s = ImageEnhance.Color(img_rgb)
        sat_factor = round(random.uniform(0.40, 2.00), 2)
        img_sat = enhancer_s.enhance(sat_factor)
        
        # Color matrix hue rotate
        h_shift = random.randint(-40, 40)
        arr_hsv = np.array(img_sat, dtype=np.float32)
        # Random channel shift
        channel = random.choice([0, 1, 2])
        arr_hsv[:, :, channel] = np.clip(arr_hsv[:, :, channel] + h_shift, 0, 255)
        result = Image.fromarray(arr_hsv.astype(np.uint8), "RGB")
        params.update({"saturation_factor": sat_factor, "channel_shift": h_shift, "channel": channel})

    elif subtype == "Grayscale":
        # Convert to grayscale and back to RGB
        gray = img.convert("L")
        result = gray.convert("RGB")
        params.update({"mode": "L_to_RGB"})

    else:  # Color_Tint
        tint_type = random.choice(["Warm", "Cool", "Sepia"])
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        strength = random.uniform(0.15, 0.35)
        
        if tint_type == "Warm":
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + strength), 0, 255)  # boost R
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - strength * 0.7), 0, 255)  # cut B
        elif tint_type == "Cool":
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 + strength), 0, 255)  # boost B
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 - strength * 0.7), 0, 255)  # cut R
        else:  # Sepia
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            arr = np.dot(arr, sepia_matrix.T)
            arr = np.clip(arr, 0, 255)

        result = Image.fromarray(arr.astype(np.uint8), "RGB")
        params.update({"tint_type": tint_type, "strength": round(strength, 2)})

    return result, subtype, params


# ==========================================
# Category 3: Noise & Compression Attacks
# ==========================================
def apply_noise_compression(img: Image.Image) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Applies randomized noise, blur, or compression artifacts."""
    subtype = random.choice(["Gaussian_Noise", "Salt_And_Pepper", "JPEG_Artifacts", "Blurring"])
    params: Dict[str, Any] = {"subtype": subtype}
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    if subtype == "Gaussian_Noise":
        sigma = round(random.uniform(12.0, 42.0), 2)
        noise = np.random.normal(0, sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(noisy_arr, "RGB")
        params.update({"sigma": sigma})

    elif subtype == "Salt_And_Pepper":
        prob = round(random.uniform(0.008, 0.035), 3)
        rnd = np.random.random(arr.shape[:2])
        salt = rnd < (prob / 2.0)
        pepper = (rnd >= (prob / 2.0)) & (rnd < prob)
        noisy_arr = arr.copy()
        noisy_arr[salt] = 255
        noisy_arr[pepper] = 0
        result = Image.fromarray(noisy_arr.astype(np.uint8), "RGB")
        params.update({"noise_prob": prob})

    elif subtype == "JPEG_Artifacts":
        quality = random.randint(12, 50)
        buffer = io.BytesIO()
        img_rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        params.update({"jpeg_quality": quality})

    else:  # Blurring
        radius = round(random.uniform(1.2, 4.0), 2)
        result = img_rgb.filter(ImageFilter.GaussianBlur(radius=radius))
        params.update({"gaussian_blur_radius": radius})

    return result, subtype, params


# ==========================================
# Category 4: Content Modifications & Overlays
# ==========================================
def apply_content(img: Image.Image) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Applies content-level changes such as text watermarks, banners, or borders."""
    subtype = random.choice(["Watermark_Text", "Border_Frame", "Background_Tint"])
    params: Dict[str, Any] = {"subtype": subtype}
    w, h = img.size
    img_rgba = img.convert("RGBA")

    if subtype == "Watermark_Text":
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        words = ["COPY", "SAMPLE", "VERIFIED", "NFT_PROOF", "REPLICA", "FORGED", "AUTHENTIC"]
        text = random.choice(words)
        
        # Approximate font scaling based on resolution
        font_size = max(18, int(min(w, h) * random.uniform(0.06, 0.14)))
        try:
            # Use default or basic system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Text position: corner, center, or random
        pos_mode = random.choice(["center", "bottom_right", "top_left", "random"])
        if pos_mode == "center":
            pos = (w // 4, h // 2)
        elif pos_mode == "bottom_right":
            pos = (int(w * 0.60), int(h * 0.85))
        elif pos_mode == "top_left":
            pos = (int(w * 0.05), int(h * 0.05))
        else:
            pos = (random.randint(int(w * 0.1), int(w * 0.7)), random.randint(int(h * 0.1), int(h * 0.7)))

        # Opacity
        alpha = random.randint(60, 200)
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (255, 215, 0)]
        base_color = random.choice(colors)
        text_color = base_color + (alpha,)

        draw.text(pos, text, fill=text_color, font=font)
        result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        params.update({"text": text, "position": pos_mode, "alpha": alpha, "font_size": font_size})

    elif subtype == "Border_Frame":
        border_pct = round(random.uniform(0.02, 0.06), 3)
        bw = int(w * border_pct)
        bh = int(h * border_pct)
        
        border_colors = [
            (255, 255, 255), (0, 0, 0), (220, 20, 60), (30, 144, 255), (255, 215, 0)
        ]
        chosen_color = random.choice(border_colors)
        
        draw = ImageDraw.Draw(img_rgba)
        # Draw frame rectangle borders
        draw.rectangle([(0, 0), (w, bh)], fill=chosen_color + (255,))
        draw.rectangle([(0, h - bh), (w, h)], fill=chosen_color + (255,))
        draw.rectangle([(0, 0), (bw, h)], fill=chosen_color + (255,))
        draw.rectangle([(w - bw, 0), (w, h)], fill=chosen_color + (255,))
        
        result = img_rgba.convert("RGB")
        params.update({"border_width_px": bw, "border_color": chosen_color})

    else:  # Background_Tint (Modifying edge / corner pixels)
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        # Sample corner pixels to detect dominant background color
        corner_pixels = [arr[5, 5], arr[5, -5], arr[-5, 5], arr[-5, -5]]
        bg_mean = np.mean(corner_pixels, axis=0)

        # Apply a localized vignette or color mask to the outer region
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist_from_center / max_dist

        # Apply tint more heavily to edges/background
        mask = np.clip(norm_dist ** 2, 0, 1)[:, :, None]
        tint_color = np.array([random.randint(50, 220) for _ in range(3)], dtype=np.float32)
        blended = arr.astype(np.float32) * (1.0 - 0.45 * mask) + tint_color * (0.45 * mask)
        result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
        params.update({"tint_color": tint_color.tolist()})

    return result, subtype, params


# ==========================================
# Main Dataset Generation Orchestrator
# ==========================================
def run_generation():
    """Generates 2,000 synthetic duplicates (500 per category) and logs ground-truth manifest."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Ensure output directories exist
    for dir_path in CATEGORY_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # 2. Gather original files by collection
    #    Creates a dict mapping collection name to list of paths to original images.
    collection_files: Dict[str, List[Path]] = {}
    for col_name, col_dir in COLLECTIONS_DIRS.items():
        if not col_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {col_dir}")
        files = sorted(list(col_dir.glob("*.png")))
        if not files:
            raise ValueError(f"No PNG files found in: {col_dir}")
        collection_files[col_name] = files
        print(f"Loaded {len(files)} originals from {col_name}")

    categories = [
        ("geometric transformation", apply_geometric, CATEGORY_DIRS["Geometric"]),
        ("color modification", apply_color, CATEGORY_DIRS["Color"]),
        ("noise & compression", apply_noise_compression, CATEGORY_DIRS["Noise_Compression"]),
        ("content modification", apply_content, CATEGORY_DIRS["Content"])
    ]

    manifest_rows: List[Dict[str, Any]] = []

    for cat_name, transform_fn, output_dir in categories:
        print(f"\n==========================================")
        print(f"Generating 500 duplicates for category: {cat_name}")
        print(f"Destination: {output_dir}")
        print(f"==========================================")

        # Stratified sampling: 167 Azuki, 167 BAYC, 166 Lil Pudgys
        samples_per_col = {
            "Azuki": 167,
            "Bored_Ape_Yacht_Club": 167,
            "Lil_Pudgys": 166
        }

        generated_for_category = 0

        for col_name, count in samples_per_col.items():
            chosen_files = random.sample(collection_files[col_name], count)
            
            for orig_path in chosen_files:
                try:
                    with Image.open(orig_path) as orig_img:
                        dup_img, subtype, params = transform_fn(orig_img)

                        # Output naming convention: {Original_Stem}_dup_{cat_short}_{idx}.png
                        cat_short = cat_name[:3].lower()
                        out_filename = f"{orig_path.stem}_dup_{cat_short}_{generated_for_category}.png"
                        out_path = output_dir / out_filename
                        dup_img.save(out_path, format="PNG")

                        manifest_rows.append({
                            "filename": out_filename,
                            "ref_image": orig_path.name,
                            "type": cat_name,
                            "sub_type": subtype.lower(),
                            "collection": col_name,
                            "parameters": json.dumps(params),
                            "is_duplicate": 1
                        })

                        generated_for_category += 1
                        if generated_for_category % 50 == 0:
                            print(f"  [{cat_name}] {generated_for_category}/500 generated...")

                except Exception as e:
                    print(f"Error processing {orig_path.name}: {e}")

        print(f"Completed {cat_name}: {generated_for_category} images generated.")

    # 3. Save Manifest Excel and CSV
    manifest_df = pd.DataFrame(manifest_rows)
    excel_path = OUTPUT_BASE_DIR / "duplications_metadata.xlsx"
    csv_path = OUTPUT_BASE_DIR / "duplications_metadata.csv"
    root_excel_path = BASE_DIR / "duplications_metadata.xlsx"

    print(f"\nSaving manifest ({len(manifest_df)} total records)...")
    manifest_df.to_excel(excel_path, index=False)
    manifest_df.to_csv(csv_path, index=False)
    manifest_df.to_excel(root_excel_path, index=False)
    print(f"Manifest saved to:\n  - {excel_path}\n  - {csv_path}\n  - {root_excel_path}")
    print("\nGeneration pipeline complete!")


if __name__ == "__main__":
    run_generation()

