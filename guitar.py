import time
import os
import difflib
from typing import List

import cv2
import numpy as np
import pytesseract
from mss import mss

# หากใช้ Windows และต้องการกำหนด path ให้ Tesseract:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ปรับค่าตามต้องการ
WORDLIST_PATH = "wordlist.10000.txt"  # ไฟล์คำศัพท์ของคุณ
INTERVAL = 0.8  # วินาที ระหว่างการจับภาพแต่ละรอบ
CHANGE_THRESHOLD = 0.6  # ถ้า similarity < threshold => ถือว่าข้อความ "เปลี่ยน"
SHOW_DEBUG_WINDOW = False  # True เพื่อโชว์หน้าต่าง preview (debug)


# โหลด wordlist แบบเก็บในหน่วยความจำ
def load_wordlist(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wordlist not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ค้นหาแบบ contain (case-insensitive)
def search_wordlist(q: str, wordlist: List[str]) -> List[str]:
    q_low = q.lower().strip()
    if not q_low:
        return []
    matches = [w for w in wordlist if q_low in w.lower()]
    return matches


# ฟังก์ชันช่วย clean console
def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# เปรียบเทียบความเหมือนของข้อความ (1.0 = เหมือนกันเป๊ะ)
def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def ocr_image(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1.0
    if max(w, h) < 400:
        scale = 2.0
    if scale != 1.0:
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(th, config=config)
    text = text.strip()
    text = " ".join(text.split())
    return text


def main():
    wordlist = load_wordlist(WORDLIST_PATH)
    print(f"Loaded {len(wordlist)} words from {WORDLIST_PATH}")

    sct = mss()

    print("\nเลือกพื้นที่บนหน้าจอ (draw rectangle) — จะใช้ตำแหน่งนี้ทุกครั้ง")
    print("หลังจากเลือกเสร็จ กด Enter/Space เพื่อยืนยัน (หน้าต่าง preview ของ OpenCV จะปรากฏ)\n")
    time.sleep(1)

    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    r = cv2.selectROI(
        "Select ROI and press ENTER/SPACE (ESC to cancel)",
        img,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select ROI and press ENTER/SPACE (ESC to cancel)")

    x, y, w, h = r
    if w == 0 or h == 0:
        print("ROI not selected. Exiting.")
        return

    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
    prev_text = ""

    # ใช้ตัวแปรท้องถิ่นเพื่อเก็บสถานะหน้าต่าง debug (ถ้า SHOW_DEBUG_WINDOW = True เราจะแสดง)
    debug_shown = False

    try:
        while True:
            region = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
            sct_img = sct.grab(region)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # ถ้าผู้ใช้ต้องการ preview ให้ใช้ debug_shown เพื่อควบคุมหน้าต่าง
            if SHOW_DEBUG_WINDOW:
                # ถ้ายังไม่เคยแสดง ให้สร้างหน้าต่าง
                if not debug_shown:
                    cv2.namedWindow("ROI preview", cv2.WINDOW_NORMAL)
                    debug_shown = True
                cv2.imshow("ROI preview", frame)
                # อ่านคีย์แบบ non-blocking; ถ้า ESC กด ให้ปิดหน้าต่าง preview เท่านั้น
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyWindow("ROI preview")
                    debug_shown = False

            text = ocr_image(frame)
            sim = similarity(prev_text, text) if prev_text else 0.0
            if not prev_text or sim < CHANGE_THRESHOLD:
                prev_text = text
                if text:
                    matches = search_wordlist(text, wordlist)
                else:
                    matches = []

                clear_console()
                print(f"Detected text: '{text}' (similarity={sim:.2f})")
                print("-" * 40)
                if matches:
                    print(f"Found {len(matches)} matches (showing up to 200):\n")
                    for idx, m in enumerate(matches[:200], 1):
                        print(f"{idx}. {m}")
                    if len(matches) > 200:
                        print(f"... and {len(matches) - 200} more")
                else:
                    print("No match found.")
                print("\n(Press Ctrl+C to quit)")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nExiting... Bye!")
    finally:
        # ทำความสะอาดหน้าต่างถ้ายังเปิดอยู่
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
