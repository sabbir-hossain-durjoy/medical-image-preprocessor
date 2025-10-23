# 🔬 Medical Image Preprocessing Tool

A professional Streamlit web application for preprocessing medical images (X-rays, CT scans) using 6 different enhancement techniques.

## 🌟 Features

### 6 Preprocessing Methods:

1. **Text Removal (Inpainting)** - Removes bright text/annotations from images
2. **BPDFHE** - Brightness Preserving Dynamic Fuzzy Histogram Equalization
3. **CLAHE + Constant + Gamma + Histogram** - 4-step enhancement pipeline
4. **CLAHE + Gamma Correction** - Simple 2-step enhancement
5. **CLAHE-YCrCb + LBP** - Texture extraction with Local Binary Patterns
6. **Histogram Equalization** - Basic global contrast enhancement

### Features:
- ✅ Upload images (PNG, JPG, JPEG, TIFF)
- ✅ Side-by-side before/after comparison
- ✅ Download processed results
- ✅ Clean, responsive UI
- ✅ Real-time processing

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Open in Browser
The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Push to GitHub
1. Create a new GitHub repository
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file path: `app.py`
6. Click "Deploy"

### Step 3: Get Your URL
Your app will be live at:
```
https://YOUR_APP_NAME.streamlit.app
```

🎉 **That's it! Share the URL with anyone!**

## 📦 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── BPDFHE.ipynb       # Original BPDFHE notebook
├── CLAHE_*.ipynb      # Original preprocessing notebooks
└── text_romove_final.py  # Original text removal script
```

## 🛠️ Technologies Used

- **Streamlit** - Web framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **scikit-image** - LBP texture features
- **Pillow** - Image I/O

## 📝 Method Descriptions

### 1. Text Removal
Uses threshold detection and inpainting to remove bright annotations from medical images.

### 2. BPDFHE
Advanced algorithm using fuzzy logic membership functions to enhance contrast while preserving brightness. Best for chest X-rays.

### 3. CLAHE + Constant + Gamma + Histogram
Complete 4-step pipeline:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Constant adjustment (brightness/contrast)
- Gamma correction (non-linear adjustment)
- Final histogram equalization

### 4. CLAHE + Gamma
Simplified 2-step enhancement - good balance between quality and speed.

### 5. CLAHE-YCrCb + LBP
Combines CLAHE in YCrCb color space with Local Binary Pattern texture extraction. Useful for feature extraction in ML models.

### 6. Histogram Equalization
Classic global histogram equalization for basic contrast enhancement.

## 🔧 Configuration

### Adjustable Parameters (in code):

**Text Removal:**
- `thresh_val=200` - Threshold for bright text detection
- `dilate_kernel=(5,5)` - Dilation kernel size
- `inpaint_radius=3` - Inpainting radius

**BPDFHE:**
- `blend=0.6` - Blending ratio
- `ksize=11` - Smoothing kernel size
- `p_low=1.0, p_high=99.0` - Percentile normalization

**CLAHE Methods:**
- `clipLimit=3.0` - CLAHE clip limit
- `tileGridSize=(8,8)` - Grid size
- `gamma=1.5` - Gamma correction value

## 🐛 Troubleshooting

### Common Issues:

**"Module not found" error:**
```bash
pip install -r requirements.txt --upgrade
```

**OpenCV import error:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless
```

**Streamlit Cloud deployment fails:**
- Make sure `requirements.txt` is in the root directory
- Use `opencv-python-headless` (not `opencv-python`)
- Check Python version compatibility (3.8-3.11)

## 📄 License

MIT License - Feel free to use and modify!

## 👨‍💻 Author

Built with ❤️ for medical image processing research

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new preprocessing methods
- Improve UI/UX
- Fix bugs
- Add documentation

## 📚 References

- CLAHE: Contrast Limited Adaptive Histogram Equalization
- BPDFHE: Brightness Preserving Dynamic Fuzzy Histogram Equalization
- LBP: Local Binary Patterns for texture classification

---

**Happy Preprocessing! 🎈**
