import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from skimage.feature import local_binary_pattern
import zipfile
import tempfile
import os
import pathlib  # kept from your original


# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Medical Image Preprocessor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================

if "last_output" not in st.session_state:
    st.session_state.last_output = None
if "last_is_bgra" not in st.session_state:
    st.session_state.last_is_bgra = False
if "last_method" not in st.session_state:
    st.session_state.last_method = None
if "method_choice" not in st.session_state:
    st.session_state.method_choice = "Text Removal"


def clear_processed_output():
    """Clear previously processed output when switching methods to avoid confusion."""
    st.session_state.last_output = None
    st.session_state.last_is_bgra = False
    st.session_state.last_method = None


# ==================== CUSTOM CSS ====================

st.markdown("""
    <style>
    .main { padding: 1rem 2rem; background: #ffffff; }

    [data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #e0e6ed; }
    [data-testid="stSidebar"] h3 { color: #1a1a1a !important; font-weight: 700; }
    [data-testid="stSidebar"] p { color: #4a5568 !important; }

    .main-header {
        background: #667eea;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .main-header h1 { color: white; font-size: 3rem; font-weight: 800; margin: 0; }
    .main-header p { color: white; font-size: 1.2rem; margin-top: 0.5rem; font-weight: 400; opacity: 0.95; }

    .image-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #e0e6ed;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .image-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.12);
    }

    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: #5568d3;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
    }

    .stDownloadButton>button {
        width: 100%;
        background: #10b981;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background: #059669;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #667eea; border-style: solid; }

    .success-banner {
        background: #10b981;
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 1.25rem 0;
    }
    .warn-banner {
        background: #f59e0b;
        color: white;
        padding: 1.1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.0rem;
        margin: 1.0rem 0;
    }

    .stMarkdown h3 { color: #1a1a1a; font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; }
    .stMarkdown h4 { color: #2d3748; font-weight: 600; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.markdown("""
    <div class="main-header">
        <h1>🔬 Medical Image Preprocessor</h1>
        <p>✨ Professional AI-powered enhancement for medical imaging ✨</p>
    </div>
""", unsafe_allow_html=True)


# ==================== UTILS ====================

def to_bgr(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA/Gray numpy image to OpenCV BGR."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


def hex_to_bgr(hex_color: str):
    """#RRGGBB -> (B,G,R)"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def compute_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = float((th > 0).mean())
    if white_ratio > 0.6:
        th = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(th) * 255

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(th)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def remove_background_with_mode(img_bgr: np.ndarray, bg_mode: str, custom_hex: str):
    mask = compute_foreground_mask(img_bgr)  # 255=fg, 0=bg
    fg = img_bgr.copy()

    if bg_mode == "Transparent":
        b, g, r = cv2.split(fg)
        a = mask
        out_bgra = cv2.merge((b, g, r, a))
        return out_bgra, True

    if bg_mode == "White":
        bg_color = (255, 255, 255)
    elif bg_mode == "Custom":
        bg_color = hex_to_bgr(custom_hex)
    else:
        bg_color = (0, 0, 0)

    out = np.zeros_like(fg)
    out[:] = bg_color
    out[mask == 255] = fg[mask == 255]
    return out, False


def resize_image(img_bgr: np.ndarray, target_w: int, target_h: int, mode: str) -> np.ndarray:
    if mode.startswith("Direct"):
        return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def adjust_brightness(img_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Brightness enhancement using alpha only."""
    return cv2.convertScaleAbs(img_bgr, alpha=float(alpha), beta=0)


# ==================== PREPROCESSING FUNCTIONS ====================

def remove_text(img, thresh_val=200, dilate_kernel=(5, 5), dilate_iter=2, inpaint_radius=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones(dilate_kernel, np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=dilate_iter)
    cleaned = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned


def bpdfhe_preprocessing(img):
    def _tri_fuzzy_memberships(n_bins=256):
        x = np.arange(n_bins, dtype=np.float32)
        dark = np.clip(1 - x/127.0, 0, 1)
        mid = np.maximum(1 - np.abs(x-127)/64.0, 0)
        bright = np.clip((x-128)/127.0, 0, 1)
        s = dark + mid + bright + 1e-6
        return dark/s, mid/s, bright/s

    def _smooth_hist(h, ksize=11):
        # Use a 1D Gaussian kernel and convolve to preserve histogram length.
        # Previous implementation created a ksize x ksize kernel then flattened it
        # which produced a much larger kernel (ksize**2) and shrank the result.
        k1d = cv2.getGaussianKernel(ksize, ksize/3).flatten()
        k1d = k1d / k1d.sum()
        # np.convolve with mode='same' returns an array of the same length as h
        return np.convolve(h, k1d, mode='same')

    def _find_minima(h, min_gap=32, max_regions=4):
        candidates = []
        for i in range(2, len(h)-2):
            if h[i] < h[i-1] and h[i] < h[i+1] and h[i] <= h[i-2] and h[i] <= h[i+2]:
                candidates.append(i)
        picks = []
        for c in candidates:
            if not picks or c - picks[-1] >= min_gap:
                picks.append(c)
            if len(picks) >= (max_regions - 1):
                break
        picks = [p for p in picks if 8 <= p <= 247]
        return sorted(picks)

    def _equalize_region(cdf, lo, hi):
        lo = int(lo); hi = int(hi)
        lo = max(0, lo); hi = min(255, hi)
        if lo >= hi:
            return np.arange(256, dtype=np.uint8)

        c_lo, c_hi = float(cdf[lo]), float(cdf[hi])
        den = max(c_hi - c_lo, 1e-8)

        m = np.arange(lo, hi + 1).astype(np.int32)
        mapped = (cdf[m] - c_lo) / den
        mapped = np.round(mapped * (hi - lo) + lo).astype(np.uint8)

        lut = np.arange(256, dtype=np.uint8)
        lut[lo:hi+1] = np.clip(mapped, 0, 255).astype(np.uint8)
        return lut

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p_low, p_high = 1.0, 99.0
    lo, hi = np.percentile(gray.astype(np.float32), [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    gray_uint8 = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255.0
    gray_uint8 = gray_uint8.astype(np.uint8)

    hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256]).flatten().astype(np.float32)
    f1, f2, f3 = _tri_fuzzy_memberships(256)
    fh = _smooth_hist(hist * (f1 + f2 + f3), ksize=11)

    splits = _find_minima(fh, min_gap=32, max_regions=4)

    regions = []
    last = 0
    for s in splits:
        regions.append((last, s))
        last = s + 1
    regions.append((last, 255))

    pdf = fh / (fh.sum() + 1e-8)
    cdf = np.cumsum(pdf)

    lut = np.arange(256, dtype=np.uint8)
    for (rlo, rhi) in regions:
        lut_region = _equalize_region(cdf, rlo, rhi)
        lut[rlo:rhi+1] = lut_region[rlo:rhi+1]

    y0 = gray_uint8
    y_map = cv2.LUT(y0, lut).astype(np.float32)

    m0, m1 = float(y0.mean()), float(y_map.mean())
    y_bp = np.clip(y_map + (m0 - m1), 0, 255)

    blend = 0.6
    out = (1.0 - blend) * y0.astype(np.float32) + blend * y_bp
    result = np.clip(out, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def clahe_constant_gamma_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    proc = clahe.apply(gray)

    alpha, beta = 1.2, 20
    proc = cv2.convertScaleAbs(proc, alpha=alpha, beta=beta)

    gamma = 1.5
    gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    proc = cv2.LUT(proc, gamma_table)

    proc = cv2.equalizeHist(proc)
    return cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)


def clahe_gamma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    proc = clahe.apply(gray)

    gamma = 1.5
    gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    proc = cv2.LUT(proc, gamma_table)

    return cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)


def clahe_ycrcb_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    img_bgr_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
    clahe_img = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2GRAY)

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(clahe_img, n_points, radius, method="uniform")
    lbp_normalized = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8))
    return cv2.cvtColor(lbp_normalized, cv2.COLOR_GRAY2BGR)


def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.equalizeHist(gray)
    return cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)


# ==================== SIDEBAR: METHODS ====================

st.sidebar.markdown("### ⚙️ Preprocessing Methods")
st.sidebar.markdown("---")

method_options = {
    "Text Removal": {"description": "Removes bright text/annotations using threshold detection + inpainting."},
    "BPDFHE": {"description": "Fuzzy histogram equalization with brightness preservation."},
    "CLAHE Complete": {"description": "CLAHE → contrast adjust → gamma correction → histogram equalization."},
    "CLAHE + Gamma": {"description": "Fast pipeline: CLAHE + gamma correction."},
    "CLAHE + LBP": {"description": "CLAHE + Local Binary Pattern texture map."},
    "Histogram EQ": {"description": "Basic global histogram equalization for contrast enhancement."},
    "Background Remove": {"description": "Removes background and supports black/white/custom/transparent."},
    "Resize": {"description": "Resize output to fixed dimensions."},
    "Brightness Enhancement": {"description": "Brightness enhancement using alpha only (no background removal)."},
}

st.sidebar.radio(
    "Select a method:",
    list(method_options.keys()),
    key="method_choice",
    on_change=clear_processed_output,
    label_visibility="collapsed"
)

method_choice = st.session_state.method_choice

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About This Method")
st.sidebar.info(method_options[method_choice]["description"])


# ==================== MAIN PAGE: WORKING PROCEDURE + SETTINGS ====================

procedure_text = {
    "Text Removal": "Removes text/annotations using threshold mask + inpainting.",
    "BPDFHE": "Brightness-preserving fuzzy histogram equalization.",
    "CLAHE Complete": "CLAHE → contrast adjust → gamma → histogram equalization.",
    "CLAHE + Gamma": "CLAHE → gamma correction.",
    "CLAHE + LBP": "CLAHE-style luminance enhancement → LBP texture map.",
    "Histogram EQ": "Global histogram equalization.",
    "Background Remove": "Foreground mask + background replacement.",
    "Resize": "Resize with aspect padding or direct resize.",
    "Brightness Enhancement": "Only applies brightness scaling (alpha). No background removal.",
}

st.markdown("### 🧾 Working Procedure")
st.info(procedure_text.get(method_choice, ""))

# ---- Defaults ----
resize_mode = "Keep Aspect + Pad"
resize_w, resize_h = 512, 512
bg_mode = "Black"
bg_custom_hex = "#FFFFFF"
brightness_alpha = 1.00

# Resize settings
if method_choice == "Resize":
    st.markdown("### 📐 Resize Settings")
    resize_mode = st.selectbox("Resize mode", ["Keep Aspect + Pad", "Direct Resize (may distort)"], index=0)
    preset = st.selectbox("Output size preset", ["256×256", "512×512", "1024×1024", "Custom"], index=1)
    if preset == "256×256":
        resize_w, resize_h = 256, 256
    elif preset == "512×512":
        resize_w, resize_h = 512, 512
    elif preset == "1024×1024":
        resize_w, resize_h = 1024, 1024
    else:
        c1, c2 = st.columns(2)
        with c1:
            resize_w = st.number_input("Width", min_value=32, max_value=4096, value=512, step=32)
        with c2:
            resize_h = st.number_input("Height", min_value=32, max_value=4096, value=512, step=32)

# Background settings
if method_choice == "Background Remove":
    st.markdown("### 🖼️ Background Options")
    bg_mode = st.selectbox("Background type", ["Black", "White", "Custom", "Transparent"], index=0)
    if bg_mode == "Custom":
        bg_custom_hex = st.color_picker("Pick background color", value="#FFFFFF")

# Brightness settings
if method_choice == "Brightness Enhancement":
    st.markdown("### 💡 Brightness Enhancement Settings")
    brightness_alpha = st.slider(
        "Brightness (alpha)",
        min_value=0.50,
        max_value=2.50,
        value=1.00,
        step=0.05,
        help="1.0 = no change. >1.0 brighter, <1.0 darker."
    )


# ==================== APPLY SELECTED METHOD ====================

def apply_preprocessing(img_bgr: np.ndarray, method_key: str):
    if method_key == "Text Removal":
        return remove_text(img_bgr), False, False
    if method_key == "BPDFHE":
        return bpdfhe_preprocessing(img_bgr), False, False
    if method_key == "CLAHE Complete":
        return clahe_constant_gamma_hist(img_bgr), False, False
    if method_key == "CLAHE + Gamma":
        return clahe_gamma(img_bgr), False, False
    if method_key == "CLAHE + LBP":
        return clahe_ycrcb_lbp(img_bgr), False, False
    if method_key == "Histogram EQ":
        return histogram_equalization(img_bgr), False, False
    if method_key == "Background Remove":
        out, is_bgra = remove_background_with_mode(img_bgr, bg_mode, bg_custom_hex)
        return out, is_bgra, (bg_mode == "Transparent")
    if method_key == "Resize":
        return resize_image(img_bgr, int(resize_w), int(resize_h), resize_mode), False, False
    if method_key == "Brightness Enhancement":
        # ✅ NO background removal here
        return adjust_brightness(img_bgr, float(brightness_alpha)), False, False

    return img_bgr, False, False


def outarray_to_pil(out_array: np.ndarray, is_bgra: bool) -> Image.Image:
    if is_bgra:
        rgba = cv2.cvtColor(out_array, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    rgb = cv2.cvtColor(out_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ==================== MAIN UI: UPLOADS ====================

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### 📤 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        label_visibility="collapsed",
        key="single_image"
    )

with col_info:
    st.markdown("### 📋 Supported Formats")
    st.markdown("""
    <div style='background: #f7fafc; padding: 1.2rem; border-radius: 10px; border: 2px solid #e0e6ed;'>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>PNG</strong></p>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>JPG/JPEG</strong></p>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>TIFF</strong></p>
        <p style='margin: 0.8rem 0 0.5rem 0; color: #4a5568; font-size: 0.9rem;'><strong>Max size:</strong> 200MB</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== DATASET ZIP PROCESSING ====================

st.markdown("---")
st.markdown("### 📦 Dataset Processing (ZIP)")

dataset_zip = st.file_uploader(
    "Upload a ZIP file containing images (PNG/JPG/TIFF). Folder structure will be preserved.",
    type=["zip"],
    help="ZIP file with PNG, JPG, JPEG, TIF, or TIFF images.",
    key="dataset_zip"
)

if dataset_zip is not None:
    st.markdown("""
        <div class="warn-banner">
            ⚠️ Large datasets may take time to process depending on size and hosting resources.
        </div>
    """, unsafe_allow_html=True)

    process_dataset = st.button("🚀 PROCESS ENTIRE DATASET (ZIP)", use_container_width=True)

    if process_dataset:
        with st.spinner("⏳ Processing dataset... Please wait"):
            supported_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            processed_count = 0
            skipped_count = 0

            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "input.zip")
                with open(zip_path, "wb") as f:
                    f.write(dataset_zip.getbuffer())

                extract_dir = os.path.join(temp_dir, "extracted")
                output_dir = os.path.join(temp_dir, "processed")
                os.makedirs(extract_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if not file.lower().endswith(supported_ext):
                            skipped_count += 1
                            continue

                        input_path = os.path.join(root, file)
                        rel_path = os.path.relpath(root, extract_dir)
                        save_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(save_dir, exist_ok=True)

                        try:
                            pil_img = Image.open(input_path)
                            img_array = np.array(pil_img)
                            img_bgr = to_bgr(img_array)

                            processed_arr, is_bgra, force_png = apply_preprocessing(img_bgr, method_choice)
                            out_pil = outarray_to_pil(processed_arr, is_bgra)

                            if force_png:
                                base, _ = os.path.splitext(file)
                                out_name = base + ".png"
                            else:
                                out_name = file

                            output_path = os.path.join(save_dir, out_name)
                            out_pil.save(output_path)
                            processed_count += 1
                        except Exception:
                            skipped_count += 1

                output_zip_path = os.path.join(temp_dir, "processed_dataset.zip")
                with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, output_dir)
                            zipf.write(full_path, arcname)

                with open(output_zip_path, "rb") as f:
                    zip_bytes = f.read()

        st.markdown(f"""
            <div class="success-banner">
                ✅ Dataset processing complete! {processed_count} images processed. {skipped_count} files skipped.
            </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇️ DOWNLOAD PROCESSED DATASET (ZIP)",
            data=zip_bytes,
            file_name=f"processed_{method_choice.replace(' ', '_')}_dataset.zip",
            mime="application/zip",
            use_container_width=True
        )


# ==================== SINGLE IMAGE FLOW ====================

if uploaded_file is not None:
    st.markdown("---")

    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = to_bgr(img_array)

    st.markdown("### 🖼️ Image Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown("#### 📸 Original Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    process_button = st.button("🚀 PROCESS IMAGE NOW", type="primary", use_container_width=True)

    if process_button:
        with st.spinner("⏳ Processing your image... Please wait"):
            processed_arr, is_bgra, _force_png = apply_preprocessing(img_bgr, method_choice)

            st.session_state.last_output = processed_arr
            st.session_state.last_is_bgra = is_bgra
            st.session_state.last_method = method_choice

    # ✅ Show processed output ONLY if it belongs to current method
    if st.session_state.last_output is not None and st.session_state.last_method == method_choice:
        processed_pil = outarray_to_pil(st.session_state.last_output, st.session_state.last_is_bgra)

        with col2:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.markdown(f"#### ✨ Processed Image ({st.session_state.last_method})")
            st.image(processed_pil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        buf = io.BytesIO()
        processed_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        base_name = os.path.splitext(uploaded_file.name)[0]
        st.download_button(
            label="⬇️ DOWNLOAD PROCESSED IMAGE (PNG)",
            data=byte_im,
            file_name=f"processed_{st.session_state.last_method.replace(' ', '_')}_{base_name}.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 3.0rem; background: #ffffff; border-radius: 16px; border: 2px solid #e0e6ed;'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>👋 Welcome!</h2>
            <p style='font-size: 1.1rem; color: #2d3748; margin: 1.25rem 0;'>
                Upload a medical image above, or upload a dataset ZIP to process all images.
            </p>
        </div>
    """, unsafe_allow_html=True)
