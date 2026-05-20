# Wine Quality Web App

PyTorch + FastAPI ilə şərab keyfiyyəti sinifləndirməsi.

## Model keyfiyyəti (vacib)

Ekranda bütün siniflərin ehtimalı **~15–25%** civarındadırsa, adətən:

1. **Inference təlimlə uyğun gəlmir** — model `StandardScaler` ilə öyrədilib, amma vebdə scaler yoxdur (və ya əksinə).
2. **Sinif balanssızlığı** — `CrossEntropyLoss`-da çəki yoxdur.

Həll: layihədə **`train_model.py`** ilə eyni CSV üzərindən `model.pth` + **`scaler.pkl`** birlikdə yenidən yaradın (sinif çəkiləri + stratified split + AdamW):

```bash
python train_model.py sizin_wine.csv
# və ya qovluqda tək bir *.csv varsa:
python train_model.py
```

Sonra serveri yenidən başladın (`uvicorn ...`). `/health`-də `scaler_loaded: true` olmalıdır.

Əvvəlki notebook modelinizi saxlamaq istəyirsinizsə, təlimdə **eyni** `StandardScaler` obyektini `joblib.dump(scaler, "scaler.pkl")` ilə saxlayın və `main.py` ilə birlikdə istifadə edin.

## Quraşdırma

```bash
pip install -r requirements.txt
```

Öz təlim etdiyiniz `model.pth` faylını layihə kökünə kopyalayın. **Təlimdə `StandardScaler` istifadə olunubsa**, eyni scaler-i `scaler.pkl` kimi saxlayın və ya `python train_model.py` ilə yenidən yaradın.

## İşə salma

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Brauzerdə açın: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

`POST /predict` — JSON body ilə 11 feature.

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Cavab: `predicted_class` (0–5, UCI/Kaggle `quality` 3–8 üçün 0–5 mapping), `confidence`, `probabilities`.

## Qeyd

Kaggle `yasserh/wine-quality-dataset` CSV-ni qovluğa atıb `python train_model.py fayl.csv` işlədə bilərsiniz. Yalnız `create_scaler.py` scaler yaradır; **`train_model.py` həm model, həm scaler-i** birlikdə düzgün uyğunlaşdırır.
