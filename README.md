# 🐱🐶 Cat or Dog Image Classifier

Aplikasi klasifikasi gambar berbasis deep learning untuk membedakan antara kucing dan anjing menggunakan Transfer Learning dengan MobileNetV2.

## 📋 Deskripsi Project

Project ini mengimplementasikan model deep learning untuk klasifikasi gambar kucing dan anjing dengan akurasi tinggi (~98%). Model dibangun menggunakan TensorFlow/Keras dengan arsitektur MobileNetV2 sebagai base model, kemudian di-fine tune untuk meningkatkan performa. Aplikasi web interaktif dibuat menggunakan Streamlit untuk memudahkan penggunaan model.

## ✨ Fitur

- 🎯 Klasifikasi gambar kucing dan anjing dengan akurasi ~98%
- 🖼️ Upload gambar custom atau pilih dari galeri default
- 📊 Menampilkan confidence score dan probabilitas setiap kelas
- 🚀 Interface yang user-friendly dengan Streamlit
- 📱 Responsive design yang dapat diakses di berbagai perangkat

## 🛠️ Teknologi yang Digunakan

- **Python 3.11.7**
- **TensorFlow/Keras** - Framework deep learning
- **MobileNetV2** - Pre-trained model untuk transfer learning
- **Streamlit** - Framework untuk web application
- **Pillow (PIL)** - Image processing
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization

## 📊 Performa Model

- **Training Accuracy**: ~97.80%
- **Validation Accuracy**: ~98.62%
- **Validation Loss**: 0.0408
- **Architecture**: MobileNetV2 with custom top layers
- **Image Size**: 224x224 pixels
- **Classes**: 2 (cats, dogs)

### Training Configuration

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Optimizer**: Adam (lr=1e-4 for initial, 1e-5 for fine-tuning)
- **Loss Function**: Sparse Categorical Crossentropy
- **Data Augmentation**: Random flip, rotation, and zoom
- **Training Data**: 20,000 images
- **Validation Data**: 5,000 images

## 📁 Struktur Project
```
ImageClasify_CatDog/
│
├── Image_Classify_Models.ipynb   # Notebook untuk training model
├── Image_classify.keras           # Model yang sudah di-training
├── app.py                         # Aplikasi Streamlit
├── default_images/                # Folder untuk gambar default
│   ├── cat.jpg
│   └── dog.jpg
├── dataset/                       # Dataset untuk training
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
├── requirements.txt               # Dependencies
└── README.md                      # Dokumentasi project
```

## 🚀 Instalasi dan Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/ImageClasify_CatDog.git
cd ImageClasify_CatDog
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv imageClassify
```

**Aktivasi virtual environment:**

- Windows:
```bash
imageClassify\Scripts\activate
```

- macOS/Linux:
```bash
source imageClassify/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.15.0
streamlit==1.31.0
pillow==10.2.0
numpy==1.26.3
matplotlib==3.8.2
```

### 4. Download Pre-trained Model

Model yang sudah di-training (`Image_classify.keras`) harus diletakkan di root directory project. Jika ingin melatih model sendiri, jalankan notebook `Image_Classify_Models.ipynb`.

## 💻 Cara Menggunakan

### Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Menggunakan Aplikasi

1. **Pilih gambar**: 
   - Pilih dari dropdown gambar default yang tersedia, atau
   - Pilih "-- Upload (pilih file) --" untuk upload gambar sendiri

2. **Upload gambar** (jika memilih opsi upload):
   - Klik tombol "Browse files"
   - Pilih file gambar (JPG, JPEG, atau PNG)

3. **Lihat hasil**:
   - Model akan otomatis memprediksi apakah gambar tersebut kucing atau anjing
   - Confidence score akan ditampilkan
   - Probabilitas untuk setiap kelas akan ditampilkan

### Training Model Sendiri

Jika ingin melatih model dari awal:

1. Siapkan dataset dengan struktur:
```
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── validation/
    ├── cats/
    └── dogs/
```

2. Buka dan jalankan `Image_Classify_Models.ipynb`

3. Model akan disimpan sebagai `Image_classify.keras`

## 📖 Penjelasan Kode

### Model Architecture
```python
# Base model: MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)
```

### Data Augmentation
```python
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### Prediction Function
```python
def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence
```

## 🎯 Hasil Training

Training dilakukan dalam 2 tahap:

**Tahap 1: Transfer Learning (10 epochs)**
- Base model di-freeze
- Training hanya pada layer tambahan
- Validation Accuracy: ~98.74%

**Tahap 2: Fine-tuning (5 epochs)**
- Base model di-unfreeze
- Training dengan learning rate lebih kecil
- Final Validation Accuracy: ~98.62%

### Training History

![Training Accuracy](docs/training_accuracy.png)
![Training Loss](docs/training_loss.png)

## 🔧 Konfigurasi

Untuk mengubah konfigurasi model atau aplikasi, edit variabel berikut:

**app.py:**
```python
MODEL_PATH = "Image_classify.keras"  # Path ke model
IMG_HEIGHT = 224                      # Tinggi gambar input
IMG_WIDTH = 224                       # Lebar gambar input
CLASS_NAMES = ['cats', 'dogs']        # Nama kelas
DEFAULT_DIR = "default_images"        # Folder gambar default
```

## 🐛 Troubleshooting

### Error: "Model file not found"
- Pastikan file `Image_classify.keras` ada di directory yang benar
- Update `MODEL_PATH` di `app.py` sesuai lokasi model

### Error: "Out of memory"
- Kurangi batch size saat training
- Gunakan gambar dengan resolusi lebih kecil

### Error: "Module not found"
- Pastikan semua dependencies sudah terinstall: `pip install -r requirements.txt`

### Prediksi tidak akurat
- Pastikan gambar yang diupload jelas dan tidak blur
- Gambar sebaiknya menampilkan objek (kucing/anjing) secara jelas
- Model dilatih dengan gambar 224x224, resolusi terlalu kecil/besar mungkin mempengaruhi akurasi

## 📈 Potential Improvements

- [ ] Menambah jumlah kelas (misalnya breed-specific)
- [ ] Implementasi multi-label classification
- [ ] Optimasi model untuk inference lebih cepat
- [ ] Deploy ke cloud (Heroku, AWS, GCP)
- [ ] Tambah fitur batch processing
- [ ] Implementasi API REST
- [ ] Tambah visualisasi Grad-CAM untuk explainability

## 👤 Author

**Mohammad Wisam Wiraghina**

- GitHub: [@wisamWR](https://github.com/wisamWR)
- LinkedIn: [Mohammad Wisam Wiraghina](www.linkedin.com/in/wisam-wira)

## 🙏 Acknowledgments

- Dataset dari [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- MobileNetV2 architecture dari Google Research
- TensorFlow dan Keras documentation
- Streamlit community

## 📞 Contact

Jika ada pertanyaan atau saran, silakan hubungi:
- Email: wisamwira27@gmail.com
- GitHub Issues: [Create an issue](https://github.com/wisamWR/Images_Classify_CatDog.git/issues)

---

⭐ Jika project ini bermanfaat, jangan lupa beri star!
