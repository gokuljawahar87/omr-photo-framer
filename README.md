# OMR Photo Framer

A Streamlit application for automatic group photo attendance counting (via YOLO face and person detection) and branded frame compositing for OMR Dreamers.

## 1. Why the Streamlit Cloud Error Occurred

The error:
```
File ".../ultralytics/data/base.py", line 12, in <module>
    import cv2
ImportError: ...
```
Streamlit Community Cloud runs on Debian Linux. `ultralytics` imports `cv2` (OpenCV), which requires native system OpenGL and GLib libraries (`libGL.so.1`, `libglib-2.0.so.0`). Since these were missing on Debian by default, the import crashed.

### The Fix:
Added `packages.txt` with:
```
libgl1
libglib2.0-0
```
Streamlit Cloud automatically reads this file and installs the required system packages during the build.

---

## 2. Reducing Cold Starts

1. **Pre-bundled Models**:
   - `yolov8n.pt` (~6.5 MB) and `yolov8n-face.pt` (~6.2 MB) are now present locally.
   - On container wake-up, the app no longer waits on network downloads from GitHub releases.
2. **Local Fonts**:
   - `BarlowCondensed-*.ttf` are bundled directly in the repo.

---

## 3. Can We Move This to Netlify?

**No, Netlify cannot run this.**
- Netlify is for static websites and small serverless functions (AWS Lambda).
- Netlify Functions have a strict **50 MB** zip size limit and short execution timeouts (10–26s).
- PyTorch + Ultralytics + OpenCV is over **1 GB** uncompressed, requiring a persistent Python process.

---

## 4. Alternative Hosting Options (Zero Cold Start / Better Performance)

| Platform | Cost | Cold Start | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Streamlit Community Cloud** | Free | ~30-60s after idle | Free, native GitHub auto-deploy. Fixed with `packages.txt`. |
| **Hugging Face Spaces** | Free | Much faster (~10s) | Built for ML/Streamlit/PyTorch. 16 GB RAM, 2 vCPUs free. |
| **Google Cloud Run** | Free tier / Pay-per-use | **0s** (with min-instances=1) | Full Docker control. Zero cold starts when configured with 1 warm instance (~$5-8/mo, or free within limits). |
| **Render / Railway** | Free / $5-7/mo | Fast / 0s on paid | Easy Docker/Python service deployment. |

---

## 5. Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
