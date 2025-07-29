import face_recognition
import os
import pickle
import cv2 # Import OpenCV untuk penanganan gambar yang lebih kuat
import sys 
import numpy as np

dataset_path = "waifu_dataset"
encodings = []
names = []

# Mengatur encoding output konsol ke UTF-8
# Ini akan membantu mengatasi UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

print("[INFO] Mulai proses encoding dataset waifu...")

for name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, name)
    if not os.path.isdir(folder_path):
        continue # Lewati jika bukan folder
    print(f"[INFO] Memproses waifu: {name}")

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        print(f"      [PROSES] {filename}...", end=' ')
        try:
            img_bgr = cv2.imread(image_path)

            if img_bgr is None:
                print(f"✗ Gagal memuat gambar (mungkin rusak atau tidak ada): {filename}")
                continue
            
            # Pastikan gambar adalah 8-bit (uint8)
            if img_bgr.dtype != np.uint8:
                img_bgr = img_bgr.astype(np.uint8)

            # Konversi ke RGB yang diharapkan face_recognition
            if img_bgr.ndim < 3: # Jika grayscale atau 1 channel
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            else: # Jika sudah berwarna (BGR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if img_rgb is None or img_rgb.size == 0:
                print(f"✗ Gambar kosong atau tidak valid setelah konversi: {filename}")
                continue

            # Coba resize ke ukuran standar untuk memastikan kompatibilitas
            # Meskipun face_recognition bisa handle berbagai ukuran, terkadang
            # format non-standar pada resolusi aslinya bisa jadi masalah
            # Ini opsional, bisa dihilangkan jika tidak membantu
            # img_rgb = cv2.resize(img_rgb, (0,0), fx=0.5, fy=0.5) # Contoh resize
            
            face_enc = face_recognition.face_encodings(img_rgb)
            
            if face_enc:
                encodings.append(face_enc[0])
                names.append(name)
                print("✓ Berhasil")
            else:
                print("✗ Tidak ditemukan wajah")
        except RuntimeError as rt_err:
            # Menggunakan repr() untuk memastikan string error bisa dicetak
            # dan menghindari UnicodeEncodeError pada pesan error itu sendiri
            print(f"✗ Runtime Error (Unsupported image type): {repr(rt_err)} di {filename}")
            print("   (Gambar mungkin rusak atau memiliki format yang tidak didukung secara spesifik oleh dlib)")
            # Anda mungkin ingin menghapus atau memeriksa gambar ini secara manual
        except Exception as e:
            # Menggunakan repr() juga di sini
            print(f"✗ Error tak terduga: {repr(e)} di {filename}")

with open("waifu_encodings.pickle", "wb") as f:
    pickle.dump((encodings, names), f)

print(f"[SELESAI] Total wajah berhasil di-encode: {len(encodings)}")
