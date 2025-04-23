import os
import gdown

def download_models(folder_url="https://drive.google.com/drive/u/1/folders/19qxil5Adf9YlQYlnetbDsRW8QsThNvZj", model_dir="models"):
    """
    Tải toàn bộ mô hình từ Google Drive folder về thư mục models/
    và in danh sách các file đã tải.
    """
    model_check_file = os.path.join(model_dir, "baseline_only_model.pkl")

    if os.path.exists(model_check_file):
        print("✅ Models already exist. Skip downloading.")
        print("📂 Danh sách file hiện có trong thư mục models/:")
        for file in os.listdir(model_dir):
            print("  -", file)
        return

    print("📥 Đang tải toàn bộ mô hình từ Google Drive...")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        gdown.download_folder(
            url=folder_url,
            output=model_dir,
            quiet=False,
            use_cookies=False
        )
        print("\n✅ Tải thành công toàn bộ mô hình!")

        # In danh sách file sau khi tải xong
        print("📦 Danh sách file đã tải về:")
        for file in os.listdir(model_dir):
            print("  -", file)

    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")

if __name__ == "__main__":
    download_models()
