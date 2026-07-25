import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from io import BytesIO
import numpy as np
import urllib.request
import os
from ultralytics import YOLO

st.title("OMR Dreamers Photo Framer")

uploaded_file = st.file_uploader("Upload Group Photo", type=["jpg", "jpeg", "png"])
session_name  = st.text_input("Session Name", placeholder="e.g. Beach Game Session")

# New frame measurements
FRAME_TOP    = 150
FRAME_BOTTOM = 161

# -------------------------
# Auto-download models
# -------------------------
def download_if_needed(path, url, min_size_mb=1):
    file_ok = os.path.exists(path) and os.path.getsize(path) > min_size_mb * 1024 * 1024
    if not file_ok:
        if os.path.exists(path):
            os.remove(path)
        urllib.request.urlretrieve(url, path)

# ✅ Correct
with st.spinner("Loading models... (first run may take a minute)"):
    download_if_needed(
        "yolov8n-face.pt",
        "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
    )
    download_if_needed(
        "yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    )
    download_if_needed(
        "BarlowCondensed-ExtraBold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-ExtraBold.ttf",
        min_size_mb=0.05
    )
    download_if_needed(
        "BarlowCondensed-ExtraBoldItalic.ttf",
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-ExtraBoldItalic.ttf",
        min_size_mb=0.05
    )
    download_if_needed(
        "BarlowCondensed-Italic.ttf",
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Italic.ttf",
        min_size_mb=0.05
    )
    download_if_needed(
        "BarlowCondensed-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
        min_size_mb=0.05
    )
@st.cache_resource
def load_face_model():
    return YOLO("yolov8n-face.pt")

@st.cache_resource
def load_person_model():
    return YOLO("yolov8n.pt")

face_model   = load_face_model()
person_model = load_person_model()


# -------------------------
# NMS helpers
# -------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, iou_threshold=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_threshold]
    return kept


# -------------------------
# People zone detection
# -------------------------
def get_people_zone(image):
    img = np.array(image)
    h, w = img.shape[:2]

    small   = Image.fromarray(img).resize((640, 640))
    results = person_model(np.array(small), conf=0.25, verbose=False)
    boxes   = results[0].boxes

    min_y = h
    max_y = 0

    if boxes is not None:
        scale_y = h / 640
        for box in boxes:
            if int(box.cls[0]) == 0:
                _, by1, _, by2 = map(int, box.xyxy[0])
                min_y = min(min_y, int(by1 * scale_y))
                max_y = max(max_y, int(by2 * scale_y))

    if max_y == 0:
        min_y = int(h * 0.38)
        max_y = int(h * 0.90)

    scan_top    = max(0, min_y - 80)
    scan_bottom = min(h, max_y + 20)
    return scan_top, scan_bottom


# -------------------------
# Face detection
# -------------------------
def count_faces(image):
    img = np.array(image)
    h, w = img.shape[:2]
    all_boxes = []

    scan_top, scan_bottom = get_people_zone(image)
    st.caption(f"🔍 Scanning rows {scan_top}px → {scan_bottom}px (image height: {h}px)")

    scale_factor      = 2
    upscaled          = Image.fromarray(img).resize(
                            (w * scale_factor, h * scale_factor), Image.LANCZOS)
    img_up            = np.array(upscaled)
    scan_top_up       = scan_top    * scale_factor
    scan_bottom_up    = scan_bottom * scale_factor
    back_row_boundary = scan_top_up + int((scan_bottom_up - scan_top_up) * 0.40)
    w_up              = w * scale_factor
    tile_size         = 640
    overlap           = 160
    stride            = tile_size - overlap

    for y in range(scan_top_up, scan_bottom_up, stride):
        for x in range(0, w_up, stride):
            x2   = min(x + tile_size, w_up)
            y2   = min(y + tile_size, scan_bottom_up)
            tile = img_up[y:y2, x:x2]

            if tile.shape[0] < 20 or tile.shape[1] < 20:
                continue

            is_back_row    = y < back_row_boundary
            conf_threshold = 0.38 if is_back_row else 0.52

            results = face_model(tile, conf=conf_threshold, verbose=False)
            boxes   = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) != 0:
                        continue

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    confidence          = float(box.conf[0])
                    box_w               = bx2 - bx1
                    box_h               = by2 - by1

                    if box_w < 25 or box_h < 25:
                        continue
                    aspect = box_w / box_h if box_h > 0 else 0
                    if aspect > 1.3 or aspect < 0.55:
                        continue
                    if box_w < 40 or box_h < 40:
                        continue

                    orig_x1 = (bx1 + x) // scale_factor
                    orig_y1 = (by1 + y) // scale_factor
                    orig_x2 = (bx2 + x) // scale_factor
                    orig_y2 = (by2 + y) // scale_factor
                    orig_w  = orig_x2 - orig_x1
                    orig_h  = orig_y2 - orig_y1

                    if orig_y2 > h * 0.85:
                        continue
                    if orig_w > 250 or orig_h > 250:
                        continue
                    if orig_w < 15 or orig_h < 15:
                        continue

                    all_boxes.append((orig_x1, orig_y1, orig_x2, orig_y2, confidence))

    final_boxes = non_max_suppression(all_boxes, iou_threshold=0.35)
    faces = [(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in final_boxes]
    return len(faces), faces


# -------------------------
# Draw face boxes (preview)
# -------------------------
def draw_faces(image, faces):
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in faces:
        draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=2)
    return img


# -------------------------
# Overlay text on new frame
# -------------------------
def draw_frame_text(canvas, face_count, session_name, date):
    draw             = ImageDraw.Draw(canvas)
    frame_w, frame_h = canvas.size

    try:
        font_number = ImageFont.truetype("BarlowCondensed-ExtraBold.ttf", 90)
        font_title  = ImageFont.truetype("BarlowCondensed-ExtraBoldItalic.ttf", 58)  # ✅ italic
        font_date   = ImageFont.truetype("BarlowCondensed-Italic.ttf", 30)            # ✅ italic, smaller, not bold
    except:
        try:
            font_number = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 90)
            font_title  = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 58)
            font_date   = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 36)
        except:
            font_number = ImageFont.load_default()
            font_title  = ImageFont.load_default()
            font_date   = ImageFont.load_default()

    # ✅ All y values shifted down by 15px
    draw.text(
        (270, 863),          # ✅ was 848, now 863
        str(face_count),
        fill=(255, 255, 255),
        anchor="mm",
        font=font_number
    )

    session_display = session_name.upper() if session_name else "SESSION TITLE"
    draw.text(
        (520, 828),          # ✅ was 805, now 820
        session_display,
        fill=(10, 30, 60),
        anchor="lm",
        font=font_title
    )

    draw.text(
        (510, 890),          # ✅ was 870, now 885
        date,
        fill=(10, 30, 60),
        anchor="lm",
        font=font_date
    )

    return canvas


# -------------------------
# Main Logic
# -------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting faces... (10–20 seconds for large photos)"):
        face_count, faces = count_faces(image)

    st.markdown(f"### 👥 Attendance: {face_count}")

    if st.checkbox("Show detected faces"):
        st.image(draw_faces(image, faces))

    # -------------------------
    # Frame compositing
    # -------------------------
    frame            = Image.open("frame.png").convert("RGBA")
    frame_w, frame_h = frame.size
    photo_area_h     = frame_h - (FRAME_TOP + FRAME_BOTTOM)

    img_w, img_h = image.size
    scale        = frame_w / img_w
    new_w        = frame_w
    new_h        = int(img_h * scale)
    resized      = image.resize((new_w, new_h), Image.LANCZOS)

    if new_h > photo_area_h:
        crop_top = (new_h - photo_area_h) // 2
        resized  = resized.crop((0, crop_top, new_w, crop_top + photo_area_h))
        new_h    = photo_area_h

    canvas   = Image.new("RGB", (frame_w, frame_h), (245, 245, 245))
    y_offset = FRAME_TOP + (photo_area_h - new_h) // 2
    canvas.paste(resized, (0, y_offset))
    canvas.paste(frame, (0, 0), frame)

    # ✅ Draw text over frame
    date   = datetime.today().strftime("%d/%m/%Y")
    canvas = draw_frame_text(canvas, face_count, session_name, date)

    st.image(canvas)

    # -------------------------
    # Export
    # -------------------------
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    st.download_button(
        "Download Image",
        data=buffer.getvalue(),
        file_name="omr_dreamers_session.jpg",
        mime="image/jpeg"
    )