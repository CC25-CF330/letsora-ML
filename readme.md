### 1. Run Notebook

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan API

```bash
python app.py
```

API akan berjalan di: http://localhost:5000

## Contoh Penggunaan

### Test dengan Python

```python
import requests

# Data mahasiswa
data = {
    "age": 20,
    "gender_encoded": 1,
    "attendance_percentage": 85,
    "mental_health_rating": 7,
    "exam_score": 75
}

# Prediksi
response = requests.post("http://localhost:5000/predict", json=data)
result = response.json()

print(f"Study hours: {result['predictions']['study_hours_per_day']} jam/hari")
print(f"Sleep hours: {result['predictions']['sleep_hours']} jam/hari")
```

### Test dengan curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 20, "gender_encoded": 1, "attendance_percentage": 85}'
```

## Input Fields

| Field                 | Tipe    | Range | Deskripsi                |
| --------------------- | ------- | ----- | ------------------------ |
| age                   | number  | 17-30 | Usia mahasiswa           |
| gender_encoded        | integer | 0-1   | 0=Perempuan, 1=Laki-laki |
| attendance_percentage | number  | 0-100 | Kehadiran kuliah (%)     |
| mental_health_rating  | number  | 1-10  | Rating kesehatan mental  |
| exam_score            | number  | 0-100 | Rata-rata nilai ujian    |

*Field lainnya opsional, akan diisi otomatis jika kosong*

## Output

```json
{
  "predictions": {
    "study_hours_per_day": 4.2,
    "sleep_hours": 7.8,
    "social_media_hours": 2.1,
    "netflix_hours": 1.4
  },
  "recommendations": {
    "study_hours": {
      "recommended_hours": 4.2,
      "advice": "Waktu belajar sudah optimal"
    }
  },
  "insights": {
    "productivity_status": "Cukup Seimbang "
  }
}
```

## Endpoints

- `GET /` - Status API
- `POST /predict` - Prediksi mahasiswa
- `GET /examples` - Contoh data
- `GET /health` - Health check

## File Structure

```
project/
├── regression.ipynb           # Notebook training
├── simple_model_saver.py      # Simpan model
├── simple_model_loader.py     # Load model  
├── app.py                     # Flask API
├── requirements.txt           # Dependencies
└── saved_models/              # Model tersimpan
```

## Troubleshooting

**Error saving model:**

```python
# Fix: gunakan format .keras
best_model.save("model.keras")
```

**API tidak start:**

```bash
# Pastikan model sudah disimpan
ls saved_models/
```

**Import error:**

```bash
# Install ulang dependencies
pip install -r requirements.txt
```

Jika ada masalah:

1. Cek apakah model sudah tersimpan di folder `saved_models/`
2. Pastikan semua dependencies terinstall
3. Test dengan data contoh dari endpoint `/examples`

---
#   l e t s o r a - M L  
 