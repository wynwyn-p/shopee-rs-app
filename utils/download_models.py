import os
import gdown
import time

def download_models(model_dir="models"):
    """
    Tải từng file mô hình cần thiết từ Google Drive.
    Không dùng download_folder để tránh tải file không mong muốn.
    """

    os.makedirs(model_dir, exist_ok=True)

    # Danh sách các file cần tải và ID tương ứng từ Google Drive
    file_links = {
        "df_final_new.parquet": "1qNhT-S5gsM5FSWWxiiIrHKXUkfha7dej",
        "baseline_only_model.pkl": "1nNnUdN0QQyPy1xVp-jCy3W5RAevIXtpp",
        "dictionary_tokenized.dict": "18yUwrnKNN6LefMRQwP6ObuCUAoLWcqqL",
        "tfidf_model_gensim.pkl": "1xHsxxTvh6wZWpphd1X5268gkr4giRQ1s",
        "gensim_index_merged": "1IwcQTNrO8hHsXCna8dDzLjPp2PPIH9Qw",  # File chính .0/.1/.2 sẽ kèm theo
    }

    print("📥 Bắt đầu kiểm tra và tải từng file mô hình...")

    for filename, file_id in file_links.items():
        save_path = os.path.join(model_dir, filename)
        if not os.path.exists(save_path):
            print(f"🔽 Tải {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, save_path, quiet=False)
        else:
            print(f"✅ {filename} đã tồn tại, bỏ qua tải.")

    print("\n✅ Toàn bộ file cần thiết đã sẵn sàng!")
    time.sleep(1)

    # In danh sách file hiện có
    print("📂 Danh sách file trong models/:")
    for f in sorted(os.listdir(model_dir)):
        print("  -", f)

if __name__ == "__main__":
    download_models()
