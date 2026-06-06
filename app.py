import re
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request
from gtts import gTTS
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
MODEL_PATH = BASE_DIR / "model" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model = YOLO(str(MODEL_PATH))
ocr_engine = RapidOCR()


def clean_old_files(folder: Path, keep_latest: int = 40):
    files = sorted(folder.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep_latest:]:
        try:
            p.unlink()
        except Exception:
            pass


def clean_ocr_text(text: str) -> str:
    """
    Membersihkan karakter OCR yang sering kebaca salah.
    Contoh: separator judul '|' sering terbaca sebagai huruf I/l.
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        line = line.strip()

        # Hilangkan token OCR yang berdiri sendiri dan sering muncul sebagai separator palsu.
        # Contoh: "PC Game Pass I 3-month" -> "PC Game Pass 3-month"
        line = re.sub(r"(?<=\w)\s+[|Il]\s+(?=[\w\d])", " ", line)

        # Rapikan spasi sebelum tanda baca
        line = re.sub(r"\s+([,.!?;:])", r"\1", line)

        # Rapikan spasi ganda
        line = re.sub(r"\s{2,}", " ", line)

        # Buang baris yang cuma berisi simbol
        if re.fullmatch(r"[\W_]+", line):
            continue

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def enhance_for_ocr(img_bgr, scale=2):
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def sort_ocr_result(result):
    items = []

    if not result:
        return items

    for item in result:
        try:
            box, text, score = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue

            pts = np.array(box).astype(float)
            x_min = float(np.min(pts[:, 0]))
            y_center = float(np.mean(pts[:, 1]))

            items.append({
                "box": pts,
                "text": text,
                "score": score,
                "x": x_min,
                "yc": y_center
            })
        except Exception:
            continue

    return sorted(items, key=lambda d: (d["yc"], d["x"]))


def group_text_lines(items, y_tolerance=22):
    if not items:
        return ""

    lines = []

    for item in items:
        placed = False
        for line in lines:
            if abs(item["yc"] - line["yc"]) <= y_tolerance:
                line["items"].append(item)
                line["yc"] = (line["yc"] + item["yc"]) / 2
                placed = True
                break

        if not placed:
            lines.append({"yc": item["yc"], "items": [item]})

    lines = sorted(lines, key=lambda l: l["yc"])

    text_lines = []
    for line in lines:
        row = sorted(line["items"], key=lambda d: d["x"])
        words = [d["text"] for d in row if d["score"] >= 0.35]
        if words:
            text_lines.append(" ".join(words))

    return clean_ocr_text("\n".join(text_lines))


def read_text_full_image(img_bgr):
    """
    OCR membaca gambar penuh, bukan crop YOLO kecil.
    Hasilnya lebih stabil dan tidak terlalu terpotong.
    """
    img_ocr = enhance_for_ocr(img_bgr, scale=2)

    try:
        result, _ = ocr_engine(img_ocr)
    except Exception as e:
        print("OCR full image gagal:", e)
        return "", []

    items = sort_ocr_result(result)

    # Koordinat OCR dibagi 2 karena gambar OCR diperbesar 2x
    for item in items:
        item["box"] = item["box"] / 2
        item["x"] = item["x"] / 2
        item["yc"] = item["yc"] / 2

    text = group_text_lines(items, y_tolerance=22)
    return text, items


def run_yolo_boxes(img_path, conf_threshold=0.45):
    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        save=False,
        verbose=False
    )[0]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        score = float(box.conf[0].cpu().numpy())
        boxes.append((x1, y1, x2, y2, score))

    return boxes


def process_image(image_path, conf_threshold=0.45, show_ocr_box=True):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError("Gambar tidak terbaca.")

    h_img, w_img = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO untuk visualisasi area teks
    yolo_boxes = run_yolo_boxes(image_path, conf_threshold=conf_threshold)

    for x1, y1, x2, y2, score in yolo_boxes:
        x1 = max(0, min(w_img, x1))
        y1 = max(0, min(h_img, y1))
        x2 = max(0, min(w_img, x2))
        y2 = max(0, min(h_img, y2))

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 130), 2)
        cv2.putText(
            img_rgb,
            f"yolo {score:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 130),
            1,
            cv2.LINE_AA
        )

    # OCR baca gambar penuh
    final_text, ocr_items = read_text_full_image(img)

    # Kotak OCR biru tipis
    if show_ocr_box:
        for item in ocr_items:
            if item["score"] < 0.35:
                continue
            pts = item["box"].astype(int)
            cv2.polylines(img_rgb, [pts], isClosed=True, color=(0, 170, 255), thickness=1)

    stem = image_path.stem
    output_image = OUTPUT_DIR / f"{stem}_detected.jpg"
    cv2.imwrite(str(output_image), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    audio_path = None
    if final_text:
        audio_path = OUTPUT_DIR / f"{stem}_audio.mp3"
        tts = gTTS(text=final_text, lang="id", slow=True)
        tts.save(str(audio_path))

    return {
        "text": final_text,
        "box_count": len(yolo_boxes),
        "ocr_count": len(ocr_items),
        "output_image": output_image.name,
        "audio_file": audio_path.name if audio_path else None
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files.get("image")
        conf = float(request.form.get("conf", 0.45))
        show_ocr_box = request.form.get("show_ocr_box") == "on"

        if not file or file.filename == "":
            error = "Silakan upload gambar terlebih dahulu."
        else:
            try:
                ext = Path(file.filename).suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    raise ValueError("Format gambar harus JPG, PNG, WEBP, atau BMP.")

                filename = f"{uuid.uuid4().hex}{ext}"
                image_path = UPLOAD_DIR / filename
                file.save(str(image_path))
                uploaded_image = filename

                result = process_image(
                    image_path,
                    conf_threshold=conf,
                    show_ocr_box=show_ocr_box
                )

                clean_old_files(UPLOAD_DIR)
                clean_old_files(OUTPUT_DIR)

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        uploaded_image=uploaded_image
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
