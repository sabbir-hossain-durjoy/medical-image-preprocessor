# Updated full app.py with ZIP dataset processing option
# Based on your uploaded Streamlit app :contentReference[oaicite:0]{index=0}

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from skimage.feature import local_binary_pattern
import pathlib  # kept (was in your original)
import zipfile
import tempfile
import os


# Page config
st.set_page_config(
    page_title="Medical Image Preprocessor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean white theme
st.markdown("""
    <style>
    /* Pure white background */
    .main {
        padding: 1rem 2rem;
        background: #ffffff;
    }
    
    /* Sidebar with white background */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 2px solid #e0e6ed;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] p {
        color: #4a5568 !important;
    }
    
    /* Header with solid purple */
    .main-header {
        background: #667eea;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Card styling with borders */
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
    
    /* Solid purple buttons */
    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
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
    
    /* Green download button */
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
    
    /* File uploader with border */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        border-style: solid;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e6ed;
    }
    
    .stRadio > div:hover {
        border-color: #667eea;
        background: #eef2ff;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background: #eef2ff;
        color: #1a1a1a;
    }
    
    /* Success banner */
    .success-banner {
        background: #10b981;
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1.5rem 0;
    }

    /* Warning banner */
    .warn-banner {
        background: #f59e0b;
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.0rem;
        margin: 1.0rem 0;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e0e6ed;
    }
    
    /* Typography */
    .stMarkdown h3 {
        color: #1a1a1a;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown h4 {
        color: #2d3748;
        font-weight: 600;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1a1a1a;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Dividers */
    hr {
        border-color: #e0e6ed;
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner color */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* General text colors */
    p {
        color: #2d3748;
    }
    
    .stMarkdown {
        color: #2d3748;
    }
    
    /* White background for all containers */
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    
    [data-testid="stHeader"] {
        background: #ffffff;
    }
    
    /* Mobile menu icon and hamburger menu - make it visible on all platforms */
    button[kind="header"],
    .st-emotion-cache-1rs6u1y,
    .st-emotion-cache-r421ms,
    [data-testid="baseButton-headerNoPadding"],
    .st-emotion-cache-pkbazv,
    .st-emotion-cache-15zrgzn {
        background: #ff4444 !important;
        border: 2px solid #cc0000 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        z-index: 9999999 !important;
        position: relative !important;
        width: 40px !important;
        height: 40px !important;
        padding: 8px !important;
        margin: 8px !important;
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[kind="header"]:hover,
    .st-emotion-cache-1rs6u1y:hover,
    .st-emotion-cache-r421ms:hover,
    [data-testid="baseButton-headerNoPadding"]:hover,
    .st-emotion-cache-pkbazv:hover,
    .st-emotion-cache-15zrgzn:hover {
        background: #ff6666 !important;
        border-color: #ff0000 !important;
        box-shadow: 0 6px 16px rgba(204, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    button[kind="header"] svg,
    .st-emotion-cache-1rs6u1y svg,
    .st-emotion-cache-r421ms svg,
    [data-testid="baseButton-headerNoPadding"] svg,
    .st-emotion-cache-pkbazv svg,
    .st-emotion-cache-15zrgzn svg {
        fill: #4285f4 !important;
        stroke: #4285f4 !important;
        width: 24px !important;
        height: 24px !important;
        filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1)) !important;
    }
    
    /* Mobile hamburger menu specific styles */
    [data-testid="collapsedControl"],
    div.st-emotion-cache-pkbazv,
    .st-emotion-cache-15zrgzn,
    button.st-emotion-cache-pkbazv,
    button[data-testid="baseButton-headerNoPadding"] {
        background: #ff4444 !important;
        color: #4285f4 !important;
        z-index: 9999999 !important;
        position: relative !important;
        padding: 8px !important;
        border: 2px solid #cc0000 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 40px !important;
        min-height: 40px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    [data-testid="collapsedControl"]:hover,
    div.st-emotion-cache-pkbazv:hover,
    .st-emotion-cache-15zrgzn:hover,
    button.st-emotion-cache-pkbazv:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover {
        background: #ff6666 !important;
        border-color: #ff0000 !important;
        box-shadow: 0 6px 16px rgba(204, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    [data-testid="collapsedControl"] svg,
    div.st-emotion-cache-pkbazv svg,
    .st-emotion-cache-15zrgzn svg,
    button.st-emotion-cache-pkbazv svg,
    button[data-testid="baseButton-headerNoPadding"] svg {
        fill: #4285f4 !important;
        stroke: #4285f4 !important;
        width: 24px !important;
        height: 24px !important;
        filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1)) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Header toolbar needs high z-index */
    [data-testid="stHeader"] {
        z-index: 9999999 !important;
        position: relative !important;
        background: #ffffff !important;
    }
    
    [data-testid="stToolbar"] {
        z-index: 9999999 !important;
        position: relative !important;
        background: #ffffff !important;
    }
    
    /* Additional menu button visibility fixes for Streamlit cloud */
    .stDeployButton,
    .stToolbar button,
    [data-testid="baseButton-headerNoPadding"] {
        background: #ffffff !important;
        border: 1px solid #e0e6ed !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        padding: 8px !important;
        margin: 4px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stDeployButton:hover,
    .stToolbar button:hover,
    [data-testid="baseButton-headerNoPadding"]:hover {
        background: #f7fafc !important;
        border-color: #667eea !important;
    }
    
    /* MOBILE RESPONSIVE STYLES */
    @media screen and (max-width: 768px) {
        /* Remove ALL horizontal scrolling */
        html, body {
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
        
        .main {
            padding: 4rem 0.5rem 3rem 0.5rem !important;
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
        
        .block-container {
            padding: 1rem 0.5rem 2rem 0.5rem !important;
            max-width: 100vw !important;
            padding-top: 5rem !important;
        }
        
        /* Smaller header on mobile */
        .main-header {
            padding: 1rem !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.85rem !important;
        }
        
        /* Force all elements to fit screen */
        * {
            max-width: 100vw !important;
            box-sizing: border-box !important;
        }
        
        /* Metrics row - FORCE single row without scroll */
        div[data-testid="column"] {
            min-width: 0 !important;
            padding: 0 0.2rem !important;
        }
        
        /* Make metrics tiny to fit */
        div[data-testid="stMetric"] {
            padding: 0.3rem 0.1rem !important;
            margin: 0 !important;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 0.75rem !important;
            line-height: 1.2 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.65rem !important;
            line-height: 1.1 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        
        /* Adjust card padding */
        .image-card {
            padding: 0.75rem !important;
            margin-bottom: 0.75rem !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        
        /* Smaller buttons */
        .stButton>button {
            font-size: 0.9rem !important;
            padding: 0.6rem 1rem !important;
        }
        
        .stDownloadButton>button {
            font-size: 0.85rem !important;
            padding: 0.6rem 1rem !important;
        }
        
        /* Reduce file uploader padding */
        [data-testid="stFileUploader"] {
            padding: 0.75rem !important;
        }
        
        /* Images */
        .stImage {
            max-width: 100% !important;
        }
        
        .stImage img {
            max-width: 100% !important;
            width: 100% !important;
            height: auto !important;
        }
        
        /* Sidebar as overlay - doesn't push content */
        [data-testid="stSidebar"] {
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            width: 85% !important;
            max-width: 320px !important;
            z-index: 999998 !important;
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.2) !important;
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out !important;
        }
        
        /* When sidebar is open */
        [data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
        }
        
        /* Overlay backdrop when sidebar is open */
        [data-testid="stSidebar"]::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.5);
            z-index: -1;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        
        [data-testid="stSidebar"][aria-expanded="true"]::before {
            opacity: 1;
            pointer-events: auto;
        }
        
        /* Main content should not be pushed */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Success banner */
        .success-banner {
            font-size: 0.95rem;
            padding: 1rem;
        }
        
        /* Images full width on mobile */
        .stImage img {
            width: 100% !important;
            height: auto !important;
        }
    }
    
    /* Tablet responsive */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 1rem 1.5rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
    }
    
    /* Small mobile devices */
    @media screen and (max-width: 480px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .main-header p {
            font-size: 0.85rem;
        }
        
        .stButton>button {
            font-size: 0.9rem;
            padding: 0.6rem 1rem;
        }
        
        .image-card {
            padding: 0.75rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Animated Header
st.markdown("""
    <div class="main-header">
        <h1>🔬 Medical Image Preprocessor</h1>
        <p>✨ Professional AI-powered enhancement for medical imaging ✨</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for method selection
st.sidebar.markdown("### ⚙️ Preprocessing Methods")
st.sidebar.markdown("---")

# Method options with emojis
method_options = {
    "Text Removal": {
        "name": "1. Text Removal (Inpainting)",
        "description": "Removes bright text/annotations from images using threshold detection and inpainting.",
        "icon": "🧹",
        "color": "#FF6B6B"
    },
    "BPDFHE": {
        "name": "2. BPDFHE (Brightness Preserving)",
        "description": "Advanced fuzzy logic-based histogram equalization that preserves brightness. Best for chest X-rays.",
        "icon": "✨",
        "color": "#4ECDC4"
    },
    "CLAHE Complete": {
        "name": "3. CLAHE + Constant + Gamma + Histogram",
        "description": "4-step pipeline: CLAHE → Contrast adjustment → Gamma correction → Histogram equalization.",
        "icon": "🎨",
        "color": "#95E1D3"
    },
    "CLAHE + Gamma": {
        "name": "4. CLAHE + Gamma Correction",
        "description": "Simple 2-step enhancement with CLAHE and gamma correction. Fast and effective.",
        "icon": "⚡",
        "color": "#F38181"
    },
    "CLAHE + LBP": {
        "name": "5. CLAHE-YCrCb + LBP",
        "description": "CLAHE in YCrCb color space + Local Binary Pattern texture extraction for ML features.",
        "icon": "🔍",
        "color": "#AA96DA"
    },
    "Histogram EQ": {
        "name": "6. Histogram Equalization",
        "description": "Basic global histogram equalization for quick contrast enhancement.",
        "icon": "📊",
        "color": "#FCBAD3"
    }
}

method_choice = st.sidebar.radio(
    "Select a method:",
    list(method_options.keys()),
    label_visibility="collapsed"
)

method = method_options[method_choice]["name"]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About This Method")
st.sidebar.info(method_options[method_choice]["description"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.markdown("""
<div style='background: #f0fdf4; 
            padding: 1rem; 
            border-radius: 10px; 
            border: 2px solid #10b981;
            color: #1a1a1a;'>
<p style='margin: 0; padding: 0.3rem 0; color: #1a1a1a;'><strong>• Text Removal:</strong> Best for annotated images</p>
<p style='margin: 0; padding: 0.3rem 0; color: #1a1a1a;'><strong>• BPDFHE:</strong> Best overall quality</p>
<p style='margin: 0; padding: 0.3rem 0; color: #1a1a1a;'><strong>• CLAHE methods:</strong> Fast processing</p>
<p style='margin: 0; padding: 0.3rem 0; color: #1a1a1a;'><strong>• LBP:</strong> For feature extraction</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='font-size: 0.9rem; color: #4a5568;'>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)


# ==================== PREPROCESSING FUNCTIONS ====================

def remove_text(img, thresh_val=200, dilate_kernel=(5, 5), dilate_iter=2, inpaint_radius=3):
    """Method 1: Text Removal"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones(dilate_kernel, np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=dilate_iter)
    cleaned = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned


def bpdfhe_preprocessing(img):
    """Method 2: BPDFHE - Brightness Preserving Dynamic Fuzzy Histogram Equalization"""

    def _tri_fuzzy_memberships(n_bins=256):
        x = np.arange(n_bins, dtype=np.float32)
        dark = np.clip(1 - x/127.0, 0, 1)
        mid = np.maximum(1 - np.abs(x-127)/64.0, 0)
        bright = np.clip((x-128)/127.0, 0, 1)
        s = dark + mid + bright + 1e-6
        return dark/s, mid/s, bright/s

    def _smooth_hist(h, ksize=7):
        k = cv2.getGaussianKernel(ksize, ksize/3)
        k = (k @ k.T).flatten()
        k = k / k.sum()
        pad = ksize // 2
        hp = np.pad(h, (pad, pad), mode='reflect')
        return np.correlate(hp, k, mode='valid')

    def _find_minima(h, min_gap=32, max_regions=4):
        candidates = []
        for i in range(2, len(h)-2):
            if h[i] < h[i-1] and h[i] < h[i+1] and h[i] <= h[i-2] and h[i] <= h[i+2]:
                candidates.append(i)
        picks = []
        for c in candidates:
            if not picks or c - picks[-1] >= min_gap:
                picks.append(c)
            if len(picks) >= (max_regions-1):
                break
        picks = [p for p in picks if 8 <= p <= 247]
        return sorted(picks)

    def _equalize_region(cdf, lo, hi):
        lo = int(lo); hi = int(hi)
        lo = max(0, lo); hi = min(len(cdf) - 1, hi)
        if lo >= hi:
            return np.arange(256, dtype=np.uint8)
        c_lo, c_hi = cdf[lo], cdf[hi]
        den = max(c_hi - c_lo, 1e-8)
        m = np.arange(lo, hi + 1)
        m = np.clip(m, 0, len(cdf) - 1).astype(np.int32)
        mapped = (cdf[m] - c_lo) / den
        mapped = np.round(mapped * (hi - lo) + lo).astype(np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        lut[lo:hi+1] = np.clip(mapped, 0, 255).astype(np.uint8)
        return lut

    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Robust normalization
    p_low, p_high = 1.0, 99.0
    lo, hi = np.percentile(gray.astype(np.float32), [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    gray_uint8 = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255.0
    gray_uint8 = gray_uint8.astype(np.uint8)

    # BPDFHE algorithm
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

    for (lo, hi) in regions:
        lut_region = _equalize_region(cdf, lo, hi)
        lut[lo:hi+1] = lut_region[lo:hi+1]

    y0 = gray_uint8.astype(np.uint8)
    y_map = cv2.LUT(y0, lut).astype(np.float32)
    m0, m1 = float(y0.mean()), float(y_map.mean())
    y_bp = np.clip(y_map + (m0 - m1), 0, 255)

    blend = 0.6
    out = (1.0 - blend) * y0.astype(np.float32) + blend * y_bp
    result = np.clip(out, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def clahe_constant_gamma_hist(img):
    """Method 3: CLAHE + Constant Adjustment + Gamma + Histogram Equalization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_processed = clahe.apply(gray)

    # Step 2: Constant Adjustment
    alpha, beta = 1.2, 20
    img_processed = cv2.convertScaleAbs(img_processed, alpha=alpha, beta=beta)

    # Step 3: Gamma Correction
    gamma = 1.5
    gamma_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                            for i in np.arange(256)]).astype("uint8")
    img_processed = cv2.LUT(img_processed, gamma_table)

    # Step 4: Histogram Equalization
    img_processed = cv2.equalizeHist(img_processed)

    return cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)


def clahe_gamma(img):
    """Method 4: CLAHE + Gamma Correction"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_processed = clahe.apply(gray)

    # Step 2: Gamma Correction
    gamma = 1.5
    gamma_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                            for i in np.arange(256)]).astype("uint8")
    img_processed = cv2.LUT(img_processed, gamma_table)

    return cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)


def clahe_ycrcb_lbp(img):
    """Method 5: CLAHE-YCrCb + LBP"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Step 1: CLAHE on Y channel (YCrCb)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    img_bgr_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
    clahe_img = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2GRAY)

    # Step 2: Local Binary Pattern (LBP)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(clahe_img, n_points, radius, method="uniform")

    # Normalize LBP to 0–255
    lbp_normalized = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8))

    return cv2.cvtColor(lbp_normalized, cv2.COLOR_GRAY2BGR)


def histogram_equalization(img):
    """Method 6: Histogram Equalization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_processed = cv2.equalizeHist(gray)
    return cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)


def apply_preprocessing(img_bgr, method_name: str):
    """Apply selected preprocessing pipeline. Works for single image and dataset ZIP."""
    if "Text Removal" in method_name:
        return remove_text(img_bgr)
    elif "BPDFHE" in method_name:
        return bpdfhe_preprocessing(img_bgr)
    elif "CLAHE + Constant + Gamma + Histogram" in method_name:
        return clahe_constant_gamma_hist(img_bgr)
    elif "CLAHE + Gamma Correction" in method_name:
        return clahe_gamma(img_bgr)
    elif "CLAHE-YCrCb + LBP" in method_name:
        return clahe_ycrcb_lbp(img_bgr)
    elif "Histogram Equalization" in method_name:
        return histogram_equalization(img_bgr)
    else:
        return img_bgr


# ==================== MAIN APP ====================

# Create columns for better layout
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
    <div style='background: #f7fafc; padding: 1.2rem; border-radius: 10px; 
                border: 2px solid #e0e6ed;'>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>PNG</strong></p>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>JPG/JPEG</strong></p>
        <p style='margin: 0.5rem 0; color: #1a1a1a;'>✅ <strong>TIFF</strong></p>
        <p style='margin: 0.8rem 0 0.5rem 0; color: #4a5568; font-size: 0.9rem;'><strong>Max size:</strong> 200MB</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== NEW: DATASET ZIP UPLOAD ====================

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
            ⚠️ Large datasets may take time to process depending on size and Streamlit hosting resources.
        </div>
    """, unsafe_allow_html=True)

    process_dataset = st.button("🚀 PROCESS ENTIRE DATASET (ZIP)", use_container_width=True)

    if process_dataset:
        with st.spinner("⏳ Processing dataset... Please wait"):
            supported_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            processed_count = 0
            skipped_count = 0

            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded zip to disk
                zip_path = os.path.join(temp_dir, "input.zip")
                with open(zip_path, "wb") as f:
                    f.write(dataset_zip.getbuffer())

                extract_dir = os.path.join(temp_dir, "extracted")
                output_dir = os.path.join(temp_dir, "processed")
                os.makedirs(extract_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                # Extract
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Process
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if not file.lower().endswith(supported_ext):
                            skipped_count += 1
                            continue

                        input_path = os.path.join(root, file)

                        # Preserve relative folder structure
                        rel_path = os.path.relpath(root, extract_dir)
                        save_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(save_dir, exist_ok=True)

                        try:
                            image = Image.open(input_path)
                            img_array = np.array(image)

                            # Convert to BGR for OpenCV
                            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                            else:
                                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                            processed = apply_preprocessing(img_bgr, method)
                            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                            out_img = Image.fromarray(processed_rgb)
                            output_path = os.path.join(save_dir, file)

                            # Save using same extension where possible; PIL may choose format by extension
                            out_img.save(output_path)
                            processed_count += 1
                        except Exception:
                            skipped_count += 1

                # Create output zip
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

# ==================== SINGLE IMAGE FLOW (UNCHANGED UI, CLEANER METHOD CALL) ====================

if uploaded_file is not None:
    st.markdown("---")

    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert RGB/RGBA/Gray to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # Image info
    st.markdown("### 📊 Image Information")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Width", f"{img_array.shape[1]}px")
    with info_col2:
        st.metric("Height", f"{img_array.shape[0]}px")
    with info_col3:
        st.metric("Channels", img_array.shape[2] if len(img_array.shape) == 3 else 1)
    with info_col4:
        st.metric("Format", uploaded_file.type.split('/')[-1].upper())

    st.markdown("---")

    # Display original image in card
    st.markdown("### 🖼️ Image Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown("#### 📸 Original Image")
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Process button
    st.markdown("<br>", unsafe_allow_html=True)
    process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
    with process_col2:
        process_button = st.button("🚀 PROCESS IMAGE NOW", type="primary", use_container_width=True)

    if process_button:
        with st.spinner("⏳ Processing your image... Please wait"):
            processed = apply_preprocessing(img_bgr, method)

            # Convert back to RGB for display
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # Display processed image in card
            with col2:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.markdown("#### ✨ Enhanced Image")
                st.image(processed_rgb, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

            # Success banner
            st.markdown("""
                <div class="success-banner">
                    ✅ Processing Complete! Your enhanced image is ready for download.
                </div>
            """, unsafe_allow_html=True)

            # Convert to bytes for download
            processed_pil = Image.fromarray(processed_rgb)
            buf = io.BytesIO()
            processed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Download button in center
            download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
            with download_col2:
                st.download_button(
                    label="⬇️ DOWNLOAD ENHANCED IMAGE",
                    data=byte_im,
                    file_name=f"enhanced_{method_choice.replace(' ', '_')}_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True
                )

            # Comparison metrics
            st.markdown("---")
            st.markdown("### 📈 Enhancement Analysis")
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            original_mean = np.mean(img_array)
            processed_mean = np.mean(processed_rgb)
            original_std = np.std(img_array)
            processed_std = np.std(processed_rgb)

            with metric_col1:
                st.metric("Mean Intensity Change",
                          f"{processed_mean:.1f}",
                          f"{processed_mean - original_mean:.1f}")
            with metric_col2:
                st.metric("Contrast (Std Dev)",
                          f"{processed_std:.1f}",
                          f"{processed_std - original_std:.1f}")
            with metric_col3:
                improvement = ((processed_std - original_std) / original_std * 100) if original_std > 0 else 0
                st.metric("Improvement", f"{abs(improvement):.1f}%")

else:
    # Empty state with instructions
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 3.5rem; background: #ffffff; border-radius: 16px; 
                    border: 2px solid #e0e6ed;'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>👋 Welcome!</h2>
            <p style='font-size: 1.2rem; color: #2d3748; margin: 1.5rem 0;'>
                Get started by uploading a medical image above
            </p>
            <div style='background: #f7fafc; padding: 1.5rem; border-radius: 10px; margin-top: 1.5rem; border: 1px solid #e0e6ed;'>
                <p style='color: #1a1a1a; margin: 0.5rem 0;'>
                    📸 <strong>Supported formats:</strong> PNG, JPG, JPEG, TIFF
                </p>
                <p style='color: #1a1a1a; margin: 0.5rem 0;'>
                    🎯 <strong>Perfect for:</strong> X-rays, CT scans, and medical imaging
                </p>
                <p style='color: #1a1a1a; margin: 0.5rem 0;'>
                    ⚡ <strong>Fast processing</strong> with professional results
                </p>
                <p style='color: #1a1a1a; margin: 0.5rem 0;'>
                    📦 <strong>New:</strong> Upload ZIP to process full dataset
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

# Footer content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Social Links Row
    link_col1, link_col2, link_col3, link_col4 = st.columns(4)

    with link_col1:
        st.markdown("""
            <a href='https://github.com/sabbir-hossain-durjoy' target='_blank' style='text-decoration: none;'>
                <div style='text-align: center; background: #f7fafc; padding: 0.8rem; border-radius: 10px; 
                           border: 1px solid #e0e6ed; transition: all 0.3s;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.3rem;'>💻</div>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #1a1a1a;'>GitHub</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    with link_col2:
        st.markdown("""
            <a href='https://www.linkedin.com/in/sabbir-hossain-durjoy-9732aa379' target='_blank' style='text-decoration: none;'>
                <div style='text-align: center; background: #f7fafc; padding: 0.8rem; border-radius: 10px; 
                           border: 1px solid #e0e6ed; transition: all 0.3s;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.3rem;'>💼</div>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #0077b5;'>LinkedIn</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    with link_col3:
        st.markdown("""
            <a href='https://scholar.google.com/citations?user=kutVEGUAAAAJ&hl=en' target='_blank' style='text-decoration: none;'>
                <div style='text-align: center; background: #f7fafc; padding: 0.8rem; border-radius: 10px; 
                           border: 1px solid #e0e6ed; transition: all 0.3s;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.3rem;'>🎓</div>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #4285f4;'>Scholar</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    with link_col4:
        st.markdown("""
            <a href='mailto:hossain15-4724@diu.edu.bd' style='text-decoration: none;'>
                <div style='text-align: center; background: #f7fafc; padding: 0.8rem; border-radius: 10px; 
                           border: 1px solid #e0e6ed; transition: all 0.3s;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.3rem;'>📧</div>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #ea4335;'>Email</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; margin-top: 1.5rem;'>
            <p style='color: #a0aec0; font-size: 0.85rem;'>
                © 2025 Medical Image Preprocessor | Built with Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
