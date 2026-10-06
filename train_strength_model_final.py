"""
Çeliklerde akma dayanımı tahmini
--------------------------------
Veri dosyası: data/steelbench_core_open.csv

Model: Medyan ile eksik değer tamamlama + Random Forest regresyonu.
Değerlendirme: grade_id grupları ayrılarak yapılan train/test bölmesi.

Not: Sonuçlar araştırma/öğrenme amaçlıdır; mühendislik tasarımı veya
malzeme güvenliği kararlarında tek başına kullanılmamalıdır.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


DATA_PATH = "data/steelbench_core_open.csv"
TARGET = "yield_strength"
GROUP_COLUMN = "grade_id"

FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]


def main():
    # 1. Veriyi yükle ve hedef değeri bulunmayan satırları çıkar.
    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET, GROUP_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Veri setinde gerekli sütunlar eksik: {missing_columns}")

    df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COLUMN]

    if len(df) < 2 or groups.nunique() < 2:
        raise ValueError("Eğitim/test ayrımı için en az iki farklı grade_id gerekli.")

    # 2. Aynı grade_id değerlerinin eğitim ve testte birlikte bulunmasını önle.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    # 3. Eksik değerleri yalnızca eğitim verisinden öğrenilen medyanlarla doldur.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 4. Modeli eğit ve test kümesinde değerlendir.
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_predictions = [y_train.mean()] * len(y_test)

    def calculate_metrics(actual, predicted):
        return {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted),
        }

    baseline_metrics = calculate_metrics(y_test, baseline_predictions)
    model_metrics = calculate_metrics(y_test, predictions)

    print("\n=== VERİ VE BÖLÜNME ===")
    print(f"Hedefi ve grade_id değeri bulunan kayıt: {len(df)}")
    print(f"Eğitim kayıtları: {len(X_train)}")
    print(f"Test kayıtları: {len(X_test)}")
    print(f"Eğitimdeki farklı grade_id sayısı: {train_groups.nunique()}")
    print(f"Testteki farklı grade_id sayısı: {test_groups.nunique()}")
    print(f"Eğitim/test grade_id kesişimi: {len(set(train_groups) & set(test_groups))}")

    print("\n=== BASELINE: EĞİTİM ORTALAMASI ===")
    print(f"MAE:  {baseline_metrics['MAE']:.2f} MPa")
    print(f"RMSE: {baseline_metrics['RMSE']:.2f} MPa")
    print(f"R²:   {baseline_metrics['R2']:.3f}")

    print("\n=== RANDOM FOREST MODELİ ===")
    print(f"MAE:  {model_metrics['MAE']:.2f} MPa")
    print(f"RMSE: {model_metrics['RMSE']:.2f} MPa")
    print(f"R²:   {model_metrics['R2']:.3f}")

    # 5. Tahmin sonuçları ve hata analizi.
    # Hata = tahmin - gerçek; negatif değer modelin düşük tahmin ettiğini gösterir.
    results = pd.DataFrame(
        {
            "Gerçek akma dayanımı (MPa)": y_test,
            "Tahmin (MPa)": predictions,
        },
        index=y_test.index,
    )
    results["Hata (MPa)"] = (
        results["Tahmin (MPa)"] - results["Gerçek akma dayanımı (MPa)"]
    )
    results["Mutlak hata (MPa)"] = results["Hata (MPa)"].abs()

    print("\n=== İLK 10 TEST TAHMİNİ ===")
    print(results.head(10).round(2).to_string())

    print("\n=== HATA ÖZETİ ===")
    print(f"Ortalama işaretli hata: {results['Hata (MPa)'].mean():.2f} MPa")
    print(f"Medyan mutlak hata: {results['Mutlak hata (MPa)'].median():.2f} MPa")
    print(f"Mutlak hata <= 50 MPa: {(results['Mutlak hata (MPa)'] <= 50).sum()}")
    print(
        "50–100 MPa: "
        f"{((results['Mutlak hata (MPa)'] > 50) & (results['Mutlak hata (MPa)'] <= 100)).sum()}"
    )
    print(
        "100–200 MPa: "
        f"{((results['Mutlak hata (MPa)'] > 100) & (results['Mutlak hata (MPa)'] <= 200)).sum()}"
    )
    print(f"> 200 MPa: {(results['Mutlak hata (MPa)'] > 200).sum()}")

    # Sonuçlar orijinal satır indeksleri üzerinden birleştirilir; satır kayması olmaz.
    info_columns = [col for col in ["grade_id", "source"] if col in df.columns]
    test_info = df.loc[y_test.index, info_columns].join(results)

    print("\n=== EN BÜYÜK 10 TAHMİN HATASI ===")
    print(
        test_info.sort_values("Mutlak hata (MPa)", ascending=False)
        .head(10)
        .round(2)
        .to_string()
    )

    # 6. Modelin kullandığı özelliklerin önem sıralaması.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    importance_table = pd.DataFrame(
        {"Özellik": feature_names, "Önem": importances}
    ).sort_values("Önem", ascending=False)

    print("\n=== ÖZELLİK ÖNEMLERİ ===")
    print(importance_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # 7. Kaynaklara göre test performansı (kaynak sütunu varsa).
    if "source" in test_info.columns:
        print("\n=== VERİ KAYNAĞINA GÖRE TEST PERFORMANSI ===")
        for source_name, group in test_info.groupby("source", dropna=False):
            actual = group["Gerçek akma dayanımı (MPa)"]
            predicted = group["Tahmin (MPa)"]
            source_mae = mean_absolute_error(actual, predicted)
            source_rmse = mean_squared_error(actual, predicted) ** 0.5
            print(
                f"{source_name}: n={len(group)}, "
                f"MAE={source_mae:.2f} MPa, RMSE={source_rmse:.2f} MPa"
            )

    print("\n=== SINIRLILIK NOTU ===")
    print(
        "Sonuçlar veri setinin kapsamına ve eksik değerlerine bağlıdır. "
        "Özellik önemleri ilişki/önem göstergesidir; nedensellik kanıtlamaz. "
        "Model gerçek mühendislik kararlarında tek başına kullanılmamalıdır."
    )


if __name__ == "__main__":
    main()
