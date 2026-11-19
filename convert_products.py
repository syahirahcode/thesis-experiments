import os
import json
import pandas as pd
from pathlib import Path

CSV_INPUT_PATH = "Products - Products.csv"
OUTPUT_JSON_PATH = Path("app/core/products.json")


def season_to_code(season_name: str) -> str:
    """Mengubah nama Season visual (misal 'Spring Light') menjadi kode 'SPRING_LIGHT'."""
    if not isinstance(season_name, str) or not season_name.strip():
        return ""
    s = season_name.strip().upper().replace("-", " ").replace("_", " ")
    return "_".join(s.split())


def normalize_category(cat_str: str) -> str:
    """Mengelompokkan kategori riasan ke tipe standar: lip, blush, eye, atau complexion."""
    if not isinstance(cat_str, str):
        return "other"
    c = cat_str.strip().lower()
    if "lip" in c:
        return "lip"
    elif "blush" in c:
        return "blush"
    elif "eye" in c:
        return "eye"
    elif any(term in c for term in ["foundation", "complexion", "cushion", "powder", "concealer"]):
        return "complexion"
    return c


def main():
    csv_file = CSV_INPUT_PATH
    if not os.path.exists(csv_file):
        candidates = [f for f in os.listdir(".") if f.endswith(".csv") and "product" in f.lower()]
        if candidates:
            csv_file = candidates[0]
        else:
            csv_file = "products.csv"

    if not os.path.exists(csv_file):
        print(f"[ERROR] File CSV tidak ditemukan di folder root.")
        return

    print(f"[*] Membaca data dari '{csv_file}'...")
    df = pd.read_csv(csv_file)

    # Deteksi kolom nama file gambar
    img_col = None
    for col in df.columns:
        if "unnamed" in col.lower() or "file" in col.lower() or "png" in str(df[col].iloc[0]).lower():
            if df[col].astype(str).str.contains(r"\.(png|jpg|jpeg)", case=False, regex=True).any():
                img_col = col
                break
    
    if not img_col and len(df.columns) >= 4:
        img_col = df.columns[3]

    products_list = []

    for idx, row in df.iterrows():
        no_val = row.get("No")
        if pd.notna(no_val) and str(no_val).isdigit():
            prod_id = f"PROD_{int(no_val):03d}"
        else:
            prod_id = f"PROD_{idx+1:03d}"

        img_file = ""
        if img_col and pd.notna(row.get(img_col)):
            img_file = str(row.get(img_col)).strip()
        elif pd.notna(row.get("Nama File")):
            img_file = f"{str(row.get('Nama File')).strip()}.png"

        season_raw = str(row.get("Season", "")).strip()
        season_code = season_to_code(season_raw)

        brand = str(row.get("Brand", "")).strip() if pd.notna(row.get("Brand")) else ""
        prod_name = str(row.get("Product Name", "")).strip() if pd.notna(row.get("Product Name")) else ""
        shade = str(row.get("Shade", "")).strip() if pd.notna(row.get("Shade")) else ""

        full_name_parts = [p for p in [brand, prod_name] if p]
        full_name = " ".join(full_name_parts)
        if shade:
            full_name = f"{full_name} - {shade}" if full_name else shade

        products_list.append({
            "product_id": prod_id,
            "product_name": full_name if full_name else f"Product {prod_id}",
            "brand": brand,
            "category": normalize_category(str(row.get("Category", ""))),
            "raw_category": str(row.get("Category", "")).strip() if pd.notna(row.get("Category")) else "",
            "shade": shade,
            "target_season": season_code,  # 1 Kode String Tunggal
            "target_undertone": str(row.get("Target Undertone", "")).strip() if pd.notna(row.get("Target Undertone")) else "",
            "target_skintone": str(row.get("Target Skintone", "")).strip() if pd.notna(row.get("Target Skintone")) else "",
            "target_saturation": str(row.get("Target Saturation", "")).strip() if pd.notna(row.get("Target Saturation")) else "",
            "image_file": img_file,
            "notes": str(row.get("Notes", "")).strip() if pd.notna(row.get("Notes")) and str(row.get("Notes")).lower() != "nan" else ""
        })

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(products_list, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Berhasil memproses {len(products_list)} baris produk ke '{OUTPUT_JSON_PATH}'")


if __name__ == "__main__":
    main()