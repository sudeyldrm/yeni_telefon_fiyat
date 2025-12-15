import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_DIR, "data", "telefonlar.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "price_range_model.pkl")


def build_price_range(df: pd.DataFrame) -> pd.Series:
    """
    price_tl -> 0..3 sınıfına çevirir.
    Hızlı ve sağlam çözüm: quartile (çeyreklik) ile 4 sınıf.
    """
    # price_tl boş/0 olanları at
    df = df[df["price_tl"].notnull()]
    df = df[df["price_tl"] > 0]

    # 4 sınıf: 0,1,2,3
    # duplicates='drop' küçük datasetlerde hata riskini azaltır
    return pd.qcut(df["price_tl"], q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)


def main():
    df = pd.read_csv(CSV_PATH)

    # Temizleme
    df = df.dropna(subset=["price_tl", "ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "is_5g", "os"])
    df = df[df["price_tl"] > 0]

    # Target üret
    df = df.copy()
    df["price_range"] = build_price_range(df)

    features_num = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "is_5g"]
    features_cat = ["os"]

    X = df[features_num + features_cat]
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial"
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
