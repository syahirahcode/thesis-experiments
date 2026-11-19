"""Konversi semua laporan HTML di output/reports ke PDF A4 (multi-halaman).

Setup:
  pip install playwright
  playwright install chromium

Pemakaian:
  python html_to_pdf.py                         # semua *.html di output/reports -> PDF di folder yang sama
  python html_to_pdf.py --src output/reports --out output/pdf
  python html_to_pdf.py --force                 # konversi ulang walau PDF sudah ada & lebih baru
  python html_to_pdf.py output/reports/report_x.html   # satu file saja

Dari backend:
  from html_to_pdf import convert_folder, convert_files
  convert_folder("output/reports")
"""
import argparse
import pathlib
import sys

from playwright.sync_api import sync_playwright

DEFAULT_SRC = "outputs/reports"
WAIT_IMAGES = "Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(r => { i.onload = i.onerror = r })))"


def convert_files(files, out_dir=None, force=False):
    """Konversi daftar file HTML. Mengembalikan (berhasil, dilewati, gagal)."""
    ok, skipped, failed = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for f in map(pathlib.Path, files):
            dest = (pathlib.Path(out_dir) if out_dir else f.parent) / (f.stem + ".pdf")
            if not force and dest.exists() and dest.stat().st_mtime >= f.stat().st_mtime:
                skipped.append(dest)
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                page = browser.new_page(viewport={"width": 1000, "height": 900})
                page.goto(f.resolve().as_uri(), wait_until="networkidle")  # JavaScript dijalankan di sini
                page.evaluate(WAIT_IMAGES)
                page.pdf(path=str(dest), format="A4", print_background=True, prefer_css_page_size=True)
                page.close()
                ok.append(dest)
            except Exception as e:  # satu file gagal tidak menghentikan yang lain
                failed.append((f, str(e)))
        browser.close()
    return ok, skipped, failed


def convert_folder(src=DEFAULT_SRC, out_dir=None, force=False):
    files = sorted(pathlib.Path(src).glob("*.html"))
    return convert_files(files, out_dir, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="file HTML tertentu (kosong = semua di --src)")
    ap.add_argument("--src", default=DEFAULT_SRC, help=f"folder sumber (default: {DEFAULT_SRC})")
    ap.add_argument("--out", help="folder tujuan PDF (default: sama dengan folder sumber)")
    ap.add_argument("--force", action="store_true", help="konversi ulang semua file")
    a = ap.parse_args()

    if a.files:
        ok, skipped, failed = convert_files(a.files, a.out, a.force)
    else:
        if not pathlib.Path(a.src).is_dir():
            sys.exit(f"Folder tidak ditemukan: {a.src}")
        ok, skipped, failed = convert_folder(a.src, a.out, a.force)

    print(f"Berhasil: {len(ok)} | Dilewati (sudah ada): {len(skipped)} | Gagal: {len(failed)}")
    for d in ok:
        print("  OK   ", d)
    for f, err in failed:
        print("  GAGAL", f, "->", err)
    sys.exit(1 if failed else 0)