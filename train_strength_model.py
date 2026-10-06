
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Veriyi yükle
DATA_PATH = "data/steelbench_core_open.csv"
df = pd.read_csv(DATA_PATH)

target = "yield_strength"

# Hedef değeri bulunmayan kayıtları çıkar
df = df.dropna(subset=[target]).copy()

# 2. Model girdilerini belirle
features = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T"
]

X = df[features]
y = df[target]
groups = df["grade_id"]

# 3. Eğitim ve test verilerini sınıflara göre ayır
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 4. Eksik değerleri eğitim verisinden öğrenilen medyanlarla doldur
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            features
        )
    ]
)

# 5. İlk modeli oluştur
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

# 6. Modeli eğit
model.fit(X_train, y_train)

# 7. Test verisi üzerinde tahmin yap
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


# 8. Baseline: Her kayıt için eğitim ortalamasını tahmin et
baseline_predictions = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)
baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5
baseline_r2 = r2_score(
    y_test, baseline_predictions
)

print("\n=== BASELINE KARŞILAŞTIRMASI ===")
print(f"Baseline MAE:  {baseline_mae:.2f} MPa")
print(f"Baseline RMSE: {baseline_rmse:.2f} MPa")
print(f"Baseline R²:   {baseline_r2:.3f}")

print("\n=== RANDOM FOREST ===")
print(f"Model MAE:  {mae:.2f} MPa")
print(f"Model RMSE: {rmse:.2f} MPa")
print(f"Model R²:   {r2:.3f}")


# 8. Sonuçları göster
print("=== AKMA DAYANIMI TAHMİN MODELİ ===")
print("Eğitim kayıtları:", len(X_train))
print("Test kayıtları:", len(X_test))
print("Eğitim sınıfı:", groups.iloc[train_idx].nunique())
print("Test sınıfı:", groups.iloc[test_idx].nunique())

print("\n=== MODEL PERFORMANSI ===")
print(f"MAE:  {mae:.2f} MPa")
print(f"RMSE: {rmse:.2f} MPa")
print(f"R²:   {r2:.3f}")

print("\n=== ÖRNEK TAHMİNLER ===")
results = pd.DataFrame({
    "Gerçek akma dayanımı (MPa)": y_test.to_numpy(),
    "Tahmin (MPa)": predictions
})

results["Mutlak hata (MPa)"] = (
    results["Gerçek akma dayanımı (MPa)"]
    - results["Tahmin (MPa)"]
).abs()

print(results.head(10).round(2).to_string(index=False))

# 9. Özellik önemlerini incele
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "regressor"
].feature_importances_

importance_table = pd.DataFrame({
    "Özellik": feature_names,
    "Önem": importances
}).sort_values("Önem", ascending=False)

print("\n=== ÖZELLİK ÖNEMLERİ ===")
print(importance_table.to_string(index=False, float_format="%.4f".__mod__))


# === HATA ANALİZİ ===

results["Hata (MPa)"] = (
    results["Tahmin (MPa)"]
    - results["Gerçek akma dayanımı (MPa)"]
)

results["Mutlak hata (MPa)"] = results["Hata (MPa)"].abs()

# En büyük hataya sahip 10 tahmin
worst_predictions = results.sort_values(
    "Mutlak hata (MPa)", ascending=False
).head(10)

print("\n=== EN BÜYÜK 10 TAHMİN HATASI ===")
print(
    worst_predictions[
        [
            "Gerçek akma dayanımı (MPa)",
            "Tahmin (MPa)",
            "Hata (MPa)",
            "Mutlak hata (MPa)"
        ]
    ].round(2).to_string(index=False)
)

# Hataların genel dağılımı
print("\n=== HATA İSTATİSTİKLERİ ===")
print(f"Ortalama hata: {results['Hata (MPa)'].mean():.2f} MPa")
print(f"En küçük hata: {results['Hata (MPa)'].min():.2f} MPa")
print(f"En büyük hata: {results['Hata (MPa)'].max():.2f} MPa")
print(f"Medyan mutlak hata: {results['Mutlak hata (MPa)'].median():.2f} MPa")

# Hata büyüklüklerine göre kaç tahmin var?
print("\n=== HATA BÜYÜKLÜĞÜ DAĞILIMI ===")
print("50 MPa veya daha az:", (results["Mutlak hata (MPa)"] <= 50).sum())
print("50-100 MPa:", (
    (results["Mutlak hata (MPa)"] > 50)
    & (results["Mutlak hata (MPa)"] <= 100)
).sum())
print("100-200 MPa:", (
    (results["Mutlak hata (MPa)"] > 100)
    & (results["Mutlak hata (MPa)"] <= 200)
).sum())
print("200 MPa üzeri:", (results["Mutlak hata (MPa)"] > 200).sum())


# === HATALI NUMUNELERİN MALZEME ÖZELLİKLERİ ===

diagnostics = X_test.copy()

diagnostics["grade_id"] = groups.iloc[test_idx].to_numpy()
diagnostics["Gerçek (MPa)"] = y_test.to_numpy()
diagnostics["Tahmin (MPa)"] = predictions
diagnostics["Mutlak hata (MPa)"] = (
    diagnostics["Gerçek (MPa)"] - diagnostics["Tahmin (MPa)"]
).abs()

print("\n=== EN HATALI 10 NUMUNENİN ÖZELLİKLERİ ===")

worst = diagnostics.nlargest(10, "Mutlak hata (MPa)")

print(worst.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Eğitim ve test verilerindeki yüksek dayanımlı çelikler
print("\n=== DAYANIM DAĞILIMI ===")

print("Eğitim hedef değerleri:")
print(y_train.describe().round(2))

print("\nTest hedef değerleri:")
print(y_test.describe().round(2))

print("\n800 MPa üzerindeki örnek sayısı:")
print("Eğitim:", (y_train > 800).sum())
print("Test:", (y_test > 800).sum())


# === EKSİK VERİ ANALİZİ ===

high_strength = df[df["yield_strength"] > 800]
normal_strength = df[df["yield_strength"] <= 800]

important_features = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si",
    "austenitize_T", "temper_T"
]

print("\n=== EKSİK VERİ KARŞILAŞTIRMASI ===")
print("800 MPa üzeri örnek sayısı:", len(high_strength))
print("800 MPa ve altı örnek sayısı:", len(normal_strength))

for feature in important_features:
    high_missing = high_strength[feature].isna().mean() * 100
    normal_missing = normal_strength[feature].isna().mean() * 100

    print(
        f"{feature:16} | "
        f"Yüksek dayanım eksik: %{high_missing:.1f} | "
        f"Diğerleri eksik: %{normal_missing:.1f}"
    )

# Veri kaynağına göre yüksek dayanımlı örnekler
if "source" in df.columns:
    print("\n=== KAYNAĞA GÖRE YÜKSEK DAYANIMLI ÖRNEKLER ===")
    print(
        high_strength["source"]
        .value_counts(dropna=False)
        .to_string()
    )

# === EN BUYUK TAHMIN HATALARININ KAYNAGI ===

worst_errors = results.sort_values(
    by="Mutlak hata (MPa)",
    ascending=False
).head(15).copy()

print("\n=== EN BUYUK 15 TAHMIN HATASI ===")
print(worst_errors.to_string(index=False))

print("\n=== EN BUYUK HATALARIN EK BILGILERI ===")

# Test kümesindeki satırların bilgilerini sonuçlarla eşleştir
test_info = df.loc[X_test.index].copy()

test_info = test_info.join(results)

worst_details = test_info.sort_values(
    by="Mutlak hata (MPa)",
    ascending=False
).head(15)

columns_to_show = [
    col for col in [
        "grade_id",
        "source",
        "yield_strength",
        "Gerçek akma dayanımı (MPa)",
        "Tahmin (MPa)",
        "Mutlak hata (MPa)",
        "Hata (MPa)"
    ]
    if col in worst_details.columns
]

print(worst_details[columns_to_show].to_string(index=False))

# === EN BÜYÜK TAHMİN HATALARININ KAYNAĞI ===

# En büyük 15 mutlak hatayı seç

print("\n=== EN BÜYÜK 15 TAHMİN HATASI ===")

# Sonuç tablosunda gerçek ve tahmin sütunlarının isimlerini kontrol et
print("Sonuç tablosundaki sütunlar:", results.columns.tolist())

# === VERI KAYNAGINA GORE MODEL PERFORMANSI ===

test_info = df.loc[X_test.index].copy()
test_info = test_info.join(results)


# === DUZELTILMIS VERI KAYNAGI ANALIZI ===

# Test verilerini ve sonuçları aynı satır sırasına getir
test_info = df.loc[X_test.index].reset_index(drop=True)
results_aligned = results.reset_index(drop=True)

# İndeks yerine satır sırasına göre birleştir
test_info = pd.concat([test_info, results_aligned], axis=1)

print("\n=== VERI KAYNAGINA GORE HATA ANALIZI ===")

if "source" in test_info.columns:
    for source_name, group in test_info.groupby("source", dropna=False):
        # Eksik hata değerlerini hesaplamaya dahil etme
        valid = group.dropna(
            subset=["Hata (MPa)", "Mutlak hata (MPa)"]
        )

        if len(valid) == 0:
            print(f"\nKaynak: {source_name}")
            print("Geçerli hata verisi bulunamadı.")
            continue

        errors = valid["Hata (MPa)"]
        mae = valid["Mutlak hata (MPa)"].mean()
        rmse = (errors.pow(2).mean()) ** 0.5

        print(f"\nKaynak: {source_name}")
        print(f"Geçerli örnek sayısı: {len(valid)}")
        print(f"MAE: {mae:.2f} MPa")
        print(f"RMSE: {rmse:.2f} MPa")
else:
    print("'source' sütunu bulunamadı.")

# === KAYNAKLARA GORE VERI DAGILIMI ===

print("\n=== KAYNAKLARA GORE DAYANIM DAGILIMI ===")

for source_name, group in df.groupby("source", dropna=False):
    strength = group["yield_strength"].dropna()

    print(f"\nKaynak: {source_name}")
    print(f"Örnek sayısı: {len(strength)}")
    print(f"Ortalama dayanım: {strength.mean():.2f} MPa")
    print(f"Medyan dayanım: {strength.median():.2f} MPa")
    print(f"Minimum: {strength.min():.2f} MPa")
    print(f"Maksimum: {strength.max():.2f} MPa")
    print(f"800 MPa üzeri: {(strength > 800).sum()}")

    features = [
        "C", "Mn", "Cr", "Mo", "Ni", "Si",
        "austenitize_T", "temper_T"
    ]

    print("Özelliklerdeki eksiklik oranları:")
    for feature in features:
        missing_pct = group[feature].isna().mean() * 100
        print(f"  {feature}: %{missing_pct:.1f}")

