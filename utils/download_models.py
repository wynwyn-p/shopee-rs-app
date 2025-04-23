import os
import gdown
import time

def download_models(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    file_links = {
        "df_final_new.parquet": "1qNhT-S5gsM5FSWWxiiIrHKXUkfha7dej",   
        "baseline_only_model.pkl": "1nNnUdN0QQyPy1xVp-jCy3W5RAevIXtpp",
        "dictionary_tokenized.dict": "18yUwrnKNN6LefMRQwP6ObuCUAoLWcqqL",
        "tfidf_model_gensim.pkl": "1xHsxxTvh6wZWpphd1X5268gkr4giRQ1s",
        "gensim_index_merged": "1IwcQTNrO8hHsXCna8dDzLjPp2PPIH9Qw"
    }

    gensim_index_ids = [
        "1W8ZbaHl4D4IpCa5XT4h46k__ykyIETTR", "1WO1ES2uWzLePiqNbAkwcOEucaAz1V3fQ",
        "1SyfYidyh7vwlPDAin-rOai9vzVAeBbzS", "1Ea3z-kEfsGmksjZisb3M9LJm80BZ_FBP",
        "1HryfyJeBzjJDjDaZCli20NKUCYy6hEo9", "1zCnH95UisD9dyYofnLQqwvGxHcDbgsN0",
        "1o1SaTbiiKUeExt0c9tKakPNC61RgwAax", "11mzgXDNp9GlW5THXJFv6YQmplFIEQ73d",
        "1h6bdppk2SGTWBOqwIQqex7l1-8w9NG8k", "1VzxW8C0XaFIHmT9zorIl8F4MgtYgWAsm",
        "1fnJCIM9fPEf8NiaTAjf_3t-rzeDYx6Tk", "1CC8xJJnCevHhTtZp1WDfH_TliOnRvGbJ",
        "1hgDzRN1vdYpU-2nfaDzUsgl7C2BRhbP-", "1inRG5woAYbP6ir7WVhUxKCzNSVYKSf5i",
        "1WQPS2A34xuW5mwwYPBp6HB3Fr_spls2h", "1wpd8T0XxEZ91cbO3etHke6ze2TI6TArX",
        "1l66XpZpLMa-s9jEwmHpZI6bd9kRa2dyD", "1dyPwMIDX1eElHhvOsreRXXicxKyiVkfk",
        "1TSgaGwzjjlGNhiXsKY8AqiFyDbzcjizF", "1l9GFiuBqd-KhzkmTXyNQDDaSn8UZXyW8",
        "1OrmtSrGwI1c7yFEOzBM_59UQuC6iMsuS", "16Xaa6fS7pMnpNEyT8xOmAqD7baExOigS",
        "1TE4mEJazFcPBhWVVUzgR0FSDJStpkEQG", "1sgcTR0f8D31LRPwEVCAKTzX_1sbBmbed",
        "1_9ifzGEbS9UEeILhd8rRP90RGCOavs5P", "1bpNrI4WvBWRn0s0nqow-_qu3_Dvz7LaM",
        "1zl4n-QdR_i2WOv5o92tgGk0ywuOnMgU3", "1-jRVyfG2maGrTWRSHeoEiCDMMPiHz1db",
        "1zeUfSe917Y9fKeWToR84I2-tJ-dNY4C3", "1_4FHn5w_qFDrsCL1nJBAnTitgNJy7orc",
        "1Jli3OY9hEO1nsBHvH7Quw5vZ9yigawcO"
    ]

    gensim_index_parts = {f"gensim_index_merged.{i}": file_id for i, file_id in enumerate(gensim_index_ids)}

    print("📥 Bắt đầu tải các file...\n")

    def safe_download(name, file_id):
        save_path = os.path.join(model_dir, name)
        url = f"https://drive.google.com/uc?id={file_id}"
        if os.path.exists(save_path):
            print(f"✅ {name} đã tồn tại, bỏ qua.")
            return

        print(f"🔽 Đang tải {name} từ {url}")
        try:
            gdown.download(url, save_path, quiet=False)
        except Exception as e:
            print(f"❌ Không thể tải {name} — Kiểm tra link: {url}")
            print(f"   → Lỗi: {e}\n")

    # Tải từng file đơn
    for fname, fid in file_links.items():
        safe_download(fname, fid)

    # Tải từng phần của gensim_index_merged
    for fname, fid in gensim_index_parts.items():
        safe_download(fname, fid)

    print("\n📦 Toàn bộ file đã được kiểm tra.")
    print("📂 File hiện có trong thư mục models/:")
    for f in sorted(os.listdir(model_dir)):
        print("  -", f)

if __name__ == "__main__":
    download_models()
