#!/usr/bin/env python3
"""
SVG Compressor — уменьшает вес SVG-файлов со встроенными (base64) картинками.

Как это работает
----------------
1. Все встроенные картинки ищутся по ВСЕМУ тексту файла (data:image/...;base64,...),
   поэтому не важно, где они лежат: в <g>, <defs>, <pattern>, <switch>,
   <foreignObject>, во вложенных <svg>, в атрибутах или в CSS.
2. Размер, в котором картинка реально показывается, считается по её width/height
   с учётом transform всех родительских групп и viewBox вложенных <svg>.
   Если размер определить нельзя (картинка в <defs>/<pattern>, проценты и т.п.),
   она только пережимается, но НЕ уменьшается по разрешению — так безопаснее.
3. Картинка уменьшается до (множитель × показанный размер), после чего для неё
   выбирается самый лёгкий вариант: PNG / PNG с палитрой 256 цветов / JPEG.
   Вариант принимается, только если он достаточно похож на оригинал (PSNR).
4. Если результат не лёгче оригинала — картинка остаётся как была.

Зависимости:  pip install pillow
Запуск с окном:   python svg_compressor.py
Запуск из консоли: python svg_compressor.py файл1.svg файл2.svg --scale 2 --quality 88
"""
import argparse
import base64
import gzip
import io
import math
import os
import queue
import re
import sys
import threading
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops, ImageOps, ImageStat

Image.MAX_IMAGE_PIXELS = None  # большие картинки не считаем «бомбой»

# ----------------------------------------------------------------------------
# Поиск встроенных картинок
# ----------------------------------------------------------------------------
# base64 может быть разбит переводами строк / &#xa; / &#10; — это тоже учитываем
DATA_URI_RE = re.compile(
    r"data:image/(png|jpe?g|gif|webp|bmp);base64,"
    r"((?:[A-Za-z0-9+/=]|\s|&#[xX]?[0-9a-fA-F]+;)+)",
    re.I,
)
_JUNK_RE = re.compile(r"&#[xX]?[0-9a-fA-F]+;|\s+")

# Контексты, где размер показа картинки определить нельзя
UNKNOWN_CONTEXT = {"defs", "symbol", "pattern", "marker", "mask", "clippath",
                   "filter", "style", "font", "cursor"}
MIN_TARGET_PX = 96  # меньше этого размера иконки не ужимаем

UNITS = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0,
         "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def parse_len(v):
    if v is None:
        return None
    m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*", v)
    if not m or m.group(2).lower() not in UNITS:
        return None
    return float(m.group(1)) * UNITS[m.group(2).lower()]


def _mul(m1, m2):
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def parse_transform(t):
    M = (1, 0, 0, 1, 0, 0)
    if not t:
        return M
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", t):
        a = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        n = name.lower()
        if n == "matrix" and len(a) == 6:
            m = tuple(a)
        elif n == "scale" and a:
            m = (a[0], 0, 0, a[1] if len(a) > 1 else a[0], 0, 0)
        elif n == "skewx" and a:
            m = (1, 0, math.tan(math.radians(a[0])), 1, 0, 0)
        elif n == "skewy" and a:
            m = (1, math.tan(math.radians(a[0])), 0, 1, 0, 0)
        else:  # translate / rotate не меняют масштаб
            continue
        M = _mul(M, m)
    return M


def matrix_scale(m):
    """Наибольший коэффициент масштабирования матрицы."""
    return max(math.hypot(m[0], m[1]), math.hypot(m[2], m[3]))


def _norm_key(s):
    return _JUNK_RE.sub("", s)


def _merge_need(need, key, req):
    if key not in need:
        need[key] = req
    else:
        prev = need[key]
        if prev is None or req is None:
            need[key] = None
        else:
            need[key] = (max(prev[0], req[0]), max(prev[1], req[1]))


def collect_requirements(text, log=print):
    """
    Обходит дерево SVG и для каждой картинки (ключ — её base64) возвращает
    нужный размер показа в px (w, h) либо None, если размер неизвестен.
    Картинка, встречающаяся несколько раз, получает максимальный из размеров.
    """
    need = {}
    try:
        # экранируем DOCTYPE-сущности: парсеру они не нужны
        root = ET.fromstring(text.encode("utf-8", errors="surrogateescape"))
    except Exception as e:  # noqa
        log(f"  ! Не удалось разобрать XML ({e}). Размеры определяю по тегам <image>.")
        for m in re.finditer(r"<image\b[^>]*>", text):
            tag = m.group(0)
            w = re.search(r'\bwidth\s*=\s*"([^"]*)"', tag)
            h = re.search(r'\bheight\s*=\s*"([^"]*)"', tag)
            w = parse_len(w.group(1)) if w else None
            h = parse_len(h.group(1)) if h else None
            req = (w, h) if w and h else None
            for u in DATA_URI_RE.finditer(tag):
                _merge_need(need, _norm_key(u.group(2)), req)
        return need

    stack = [(root, (1, 0, 0, 1, 0, 0), False, True)]
    while stack:  # итеративно, чтобы не упереться в лимит рекурсии
        el, pm, unknown, is_root = stack.pop()
        tag = el.tag.rsplit("}", 1)[-1].lower() if isinstance(el.tag, str) else ""
        m = _mul(pm, parse_transform(el.get("transform")))
        if tag == "svg" and not is_root:
            w, h = parse_len(el.get("width")), parse_len(el.get("height"))
            vb = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", el.get("viewBox") or "")
            if w and h and len(vb) == 4 and float(vb[2]) > 0 and float(vb[3]) > 0:
                s = min(w / float(vb[2]), h / float(vb[3]))
                m = _mul(m, (s, 0, 0, s, 0, 0))
        unk = unknown or tag in UNKNOWN_CONTEXT

        req = None
        if tag in ("image", "img") and not unk:
            w, h = parse_len(el.get("width")), parse_len(el.get("height"))
            if w and h and w > 0 and h > 0:
                s = matrix_scale(m)
                req = (w * s, h * s)

        texts = list(el.attrib.values())
        if el.text:
            texts.append(el.text)
        for t in texts:
            if "data:image/" in t:
                for u in DATA_URI_RE.finditer(t):
                    _merge_need(need, _norm_key(u.group(2)), req)

        for ch in el:
            stack.append((ch, m, unk, False))
    return need


# ----------------------------------------------------------------------------
# Сжатие одной картинки
# ----------------------------------------------------------------------------
def _flat(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")


def _psnr(a, b):
    diff = ImageChops.difference(a, b)
    ss = sum(ImageStat.Stat(diff).sum2)
    mse = ss / (a.width * a.height * 3)
    return 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)


def _png(im, icc):
    buf = io.BytesIO()
    kw = {"icc_profile": icc} if icc else {}
    im.save(buf, "PNG", optimize=True, **kw)
    return buf.getvalue()


def compress_image(b64, req, scale, quality, palette):
    """Возвращает (mime, bytes) нового варианта или None, если оставить как есть."""
    raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
    im = Image.open(io.BytesIO(raw))
    if getattr(im, "n_frames", 1) > 1:
        return None  # анимация — не трогаем
    im.load()
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "RGBA"):
        has_a = "A" in im.getbands() or "transparency" in im.info
        im = im.convert("RGBA" if has_a else "RGB")

    resized = False
    if req:
        tw = max(req[0] * scale, MIN_TARGET_PX)
        th = max(req[1] * scale, MIN_TARGET_PX)
        f = max(tw / im.width, th / im.height)
        if f < 0.95:
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                           Image.Resampling.LANCZOS)
            resized = True

    has_alpha = im.mode == "RGBA" and int(im.getchannel("A").getextrema()[0]) < 255
    if im.mode == "RGBA" and not has_alpha:
        im = im.convert("RGB")

    ref = _flat(im)
    full = _png(im, icc)
    cands = [("image/png", full)]

    if palette:
        try:
            if has_alpha:
                q = im.quantize(256, Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
            else:
                q = im.quantize(256, Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            qb = _png(q, icc)
            if len(qb) < len(full) * 0.8 and _psnr(ref, _flat(Image.open(io.BytesIO(qb)))) >= 36:
                cands.append(("image/png", qb))
        except Exception:
            pass

    if not has_alpha:
        buf = io.BytesIO()
        kw = {"icc_profile": icc} if icc else {}
        im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True,
                               progressive=True, subsampling=0, **kw)
        jb = buf.getvalue()
        if _psnr(ref, Image.open(io.BytesIO(jb)).convert("RGB")) >= 34:
            cands.append(("image/jpeg", jb))

    mime, data = min(cands, key=lambda c: len(c[1]))
    limit = len(raw) if resized else len(raw) * 0.9
    return (mime, data) if len(data) < limit else None


# ----------------------------------------------------------------------------
# Сжатие целого файла
# ----------------------------------------------------------------------------
def compress_svg(src, dst, scale=2.0, quality=88, palette=True, log=print, progress=None):
    data = open(src, "rb").read()
    if src.lower().endswith(".svgz") or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    size_before = os.path.getsize(src)
    text = data.decode("utf-8", errors="surrogateescape")

    keys = list(dict.fromkeys(_norm_key(m.group(2)) for m in DATA_URI_RE.finditer(text)))
    total_refs = len(DATA_URI_RE.findall(text))
    log(f"  Найдено вложенных картинок: {total_refs} (уникальных: {len(keys)})")
    if not keys:
        log("  Вложенных картинок нет — нечего сжимать.")
        return None

    need = collect_requirements(text, log)
    cache, changed = {}, 0
    for i, key in enumerate(keys, 1):
        try:
            res = compress_image(key, need.get(key), scale, quality, palette)
        except Exception as e:  # битая картинка — оставляем как есть
            log(f"  ! Картинка {i}: пропущена ({e})")
            res = None
        if res:
            mime, buf = res
            cache[key] = f"data:{mime};base64,{base64.b64encode(buf).decode()}"
            changed += 1
        if progress:
            progress(i / len(keys))

    def repl(m):
        return cache.get(_norm_key(m.group(2)), m.group(0))

    out = DATA_URI_RE.sub(repl, text)
    with open(dst, "wb") as f:
        f.write(out.encode("utf-8", errors="surrogateescape"))
    size_after = os.path.getsize(dst)
    log(f"  Пережато: {changed} из {len(keys)}. "
        f"{size_before / 1e6:.2f} МБ → {size_after / 1e6:.2f} МБ "
        f"(в {size_before / max(size_after, 1):.1f} раза меньше)")
    return size_before, size_after


def output_path(src, out_dir=None):
    base = os.path.basename(src)
    name = re.sub(r"\.svgz?$", "", base, flags=re.I)
    return os.path.join(out_dir or os.path.dirname(os.path.abspath(src)), name + "_compressed.svg")


# ----------------------------------------------------------------------------
# Окно (tkinter)
# ----------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("SVG Compressor")
    root.geometry("720x560")
    q = queue.Queue()
    files = []

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    # --- список файлов
    top = ttk.Frame(frm)
    top.pack(fill="x")
    lb = tk.Listbox(top, height=5, selectmode="extended")
    lb.pack(side="left", fill="both", expand=True)
    btns = ttk.Frame(top)
    btns.pack(side="left", padx=(8, 0), anchor="n")

    def add_files():
        for p in filedialog.askopenfilenames(filetypes=[("SVG", "*.svg *.svgz"), ("Все файлы", "*.*")]):
            if p not in files:
                files.append(p)
                lb.insert("end", p)

    def clear_files():
        files.clear()
        lb.delete(0, "end")

    ttk.Button(btns, text="Добавить файлы…", command=add_files).pack(fill="x")
    ttk.Button(btns, text="Очистить", command=clear_files).pack(fill="x", pady=(4, 0))

    # --- настройки
    opt = ttk.LabelFrame(frm, text="Настройки", padding=8)
    opt.pack(fill="x", pady=8)
    scale_var = tk.DoubleVar(value=2.0)
    qual_var = tk.IntVar(value=88)
    pal_var = tk.BooleanVar(value=True)
    out_var = tk.StringVar()

    ttk.Label(opt, text="Разрешение картинок (× от размера на схеме):").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(opt, from_=1.0, to=4.0, increment=0.5, width=6, textvariable=scale_var).grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(opt, text="Качество JPEG (50–100):").grid(row=1, column=0, sticky="w")
    ttk.Spinbox(opt, from_=50, to=100, increment=1, width=6, textvariable=qual_var).grid(row=1, column=1, sticky="w", padx=6)
    ttk.Checkbutton(opt, text="Сокращать палитру PNG до 256 цветов, если это незаметно",
                    variable=pal_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(opt, text="Папка результата (пусто = рядом с оригиналом):").grid(row=3, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(opt, textvariable=out_var, width=30).grid(row=3, column=1, columnspan=2, sticky="we", padx=6, pady=(6, 0))
    ttk.Button(opt, text="Обзор…", command=lambda: out_var.set(filedialog.askdirectory() or out_var.get())
               ).grid(row=3, column=3, pady=(6, 0))
    opt.columnconfigure(2, weight=1)

    # --- запуск и лог
    start_btn = ttk.Button(frm, text="Сжать")
    start_btn.pack(fill="x")
    bar = ttk.Progressbar(frm, maximum=1.0)
    bar.pack(fill="x", pady=6)
    log_box = ScrolledText(frm, height=12, state="disabled")
    log_box.pack(fill="both", expand=True)

    def log(msg):
        q.put(("log", msg))

    def worker(paths, scale, quality, palette, out_dir):
        total_b = total_a = 0
        for i, p in enumerate(paths, 1):
            log(f"[{i}/{len(paths)}] {os.path.basename(p)}")
            try:
                res = compress_svg(p, output_path(p, out_dir), scale, quality, palette,
                                   log=log, progress=lambda f, i=i: q.put(("bar", f)))
                if res:
                    total_b += res[0]
                    total_a += res[1]
            except Exception as e:
                log(f"  ! Ошибка: {e}")
        if total_b:
            log(f"\nГотово. Всего: {total_b / 1e6:.2f} МБ → {total_a / 1e6:.2f} МБ")
        else:
            log("\nГотово.")
        q.put(("done", None))

    def start():
        if not files:
            messagebox.showinfo("SVG Compressor", "Сначала добавьте файлы.")
            return
        try:
            scale, quality = float(scale_var.get()), int(qual_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("SVG Compressor", "Проверьте числа в настройках.")
            return
        start_btn.config(state="disabled")
        bar["value"] = 0
        threading.Thread(target=worker, daemon=True,
                         args=(list(files), max(0.5, scale), min(100, max(30, quality)),
                               pal_var.get(), out_var.get().strip() or None)).start()

    start_btn.config(command=start)

    def poll():
        try:
            while True:
                kind, val = q.get_nowait()
                if kind == "log":
                    log_box.config(state="normal")
                    log_box.insert("end", val + "\n")
                    log_box.see("end")
                    log_box.config(state="disabled")
                elif kind == "bar":
                    bar["value"] = val
                elif kind == "done":
                    start_btn.config(state="normal")
                    bar["value"] = 1.0
        except queue.Empty:
            pass
        root.after(100, poll)

    poll()
    root.mainloop()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Сжатие SVG со встроенными картинками")
    ap.add_argument("files", nargs="*", help="SVG-файлы (без аргументов откроется окно)")
    ap.add_argument("--scale", type=float, default=2.0, help="множитель разрешения (по умолчанию 2)")
    ap.add_argument("--quality", type=int, default=88, help="качество JPEG (по умолчанию 88)")
    ap.add_argument("--no-palette", action="store_true", help="не сокращать палитру PNG")
    ap.add_argument("--out", help="папка для результата")
    a = ap.parse_args()
    if not a.files:
        run_gui()
        return
    for p in a.files:
        print(os.path.basename(p))
        compress_svg(p, output_path(p, a.out), a.scale, a.quality, not a.no_palette)


if __name__ == "__main__":
    main()
