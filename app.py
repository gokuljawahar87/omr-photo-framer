import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from io import BytesIO

st.title("OMR Dreamers Photo Framer")

uploaded_file = st.file_uploader("Upload Group Photo", type=["jpg","jpeg","png"])
session = st.number_input("Session Number", min_value=1, step=1)

FRAME_TOP = 210
FRAME_BOTTOM = 210


# -------------------------
# ordinal text
# -------------------------

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1:"st",2:"nd",3:"rd"}.get(n % 10,"th")
    return str(n) + suffix


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    frame = Image.open("frame.png").convert("RGBA")

    frame_w, frame_h = frame.size

    photo_area_h = frame_h - (FRAME_TOP + FRAME_BOTTOM)

    img_w, img_h = image.size

    # -------------------------
    # scale to frame width
    # -------------------------

    scale = frame_w / img_w

    new_w = frame_w
    new_h = int(img_h * scale)

    resized = image.resize((new_w,new_h), Image.LANCZOS)

    # vertical crop only if needed
    if new_h > photo_area_h:

        crop_top = (new_h - photo_area_h) // 2

        resized = resized.crop((0, crop_top, new_w, crop_top + photo_area_h))

        new_h = photo_area_h

    canvas = Image.new("RGB", (frame_w, frame_h), (255,255,255))

    y_offset = FRAME_TOP + (photo_area_h - new_h) // 2

    canvas.paste(resized, (0, y_offset))

    canvas.paste(frame, (0,0), frame)

    draw = ImageDraw.Draw(canvas)

    date = datetime.today().strftime("%d/%m/%Y")

    line1 = "DRHM Structured Training"
    line2 = f"{ordinal(session)} Session – {date}"

    try:
        font_big = ImageFont.truetype("Nirmala.ttf",130)
	font_small = ImageFont.truetype("Nirmala.ttf",105)
    except:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # -------------------------
    # footer text
    # -------------------------

    draw.text(
        (frame_w/2, frame_h-220),
        line1,
        fill=(0,0,0),
        anchor="mm",
        font=font_big
    )

    draw.text(
        (frame_w/2, frame_h-80),
        line2,
        fill=(0,0,0),
        anchor="mm",
        font=font_small
    )

    st.image(canvas)

    # -------------------------
    # JPEG export
    # -------------------------

    buffer = BytesIO()

    canvas.save(buffer, format="JPEG", quality=92)

    st.download_button(
        "Download Image",
        data=buffer.getvalue(),
        file_name="omr_dreamers_session.jpg",
        mime="image/jpeg"
    )