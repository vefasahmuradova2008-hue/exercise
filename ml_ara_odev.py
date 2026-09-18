"""
Amaç:
    1.Churn tahmini yapmak
Addımlar:
    1.Veri setini yüklemek
    2.Eksik değerleri kontrol etmek.
    3.Eksik değerleri silmek ve ya doldurmak
    4.Kategorik değişkenlere One-Encoding uygula
    5.Sayısal değerlere ölçekleme uygula
    6.Öznitelik üretmek
    7.Veriyi train, validation ve test kümelerine ayırmak
    8.Logistic Regression uygula
    9.KNN uygula
    10.Test seti için confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazmak
Kurulumlar: pip install sklearn, pandas,
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1.Veri setini yüklemek
df =  pd.read_csv("veri_seti.csv")
print(df.head())
print(df.info())

#2.Eksik değerleri kontrol etmek
print(df.isnull().sum())

#3.Eksik değerleri silmek ve ya doldurmak
df_dropna = df.dropna()
print(f"Eksik degerler silindikten sonra: \n{df_dropna}")

df_filled = df.copy()
df_filled["churn"] = df_filled["churn"].fillna(df_filled["churn"].mode()[0])
print(f"Eksik veriler doldurulduktan sonra: \n{df_filled}")

#4.Kategorik değişkenlere One-Encoding uygula
X = df_filled.drop(columns=["churn"])
X = pd.get_dummies(X, columns=["sehir", "uyelik_tipi", "destek_talebi_sayisi"], drop_first=True, dtype=int)
print(f"Kategorik donusum sonrasi ozellikler: \n{X}")

#5.Sayısal değerlere ölçekleme uygula
df_standard = df_filled.copy()
df_normalized = df_filled.copy()
scaler = StandardScaler() #Standardizasyon
df_standard['gelir_standard'] = scaler.fit_transform(df_filled[['gelir']])
print(f"df_standard: \n{df_standard}")

scaler = MinMaxScaler() #Normalizasyon
df_normalized['gelir_normalized'] = scaler.fit_transform(df_filled[['gelir']])
print(f"df_normalized: \n{df_normalized}")

#6.Öznitelik üretmek
df_filled["aylik_gelir"] = (df_filled["gelir"]/12).round(1)
print(df_filled.head())

#7.Veriyi train, validation ve test kümelerine ayırmak
y = df_filled["churn"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"X_val: {X_val.shape}")


#8.Logistic Regression uygula
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
log_reg = LogisticRegression(l1_ratio=0, C=1, max_iter=1000)
log_reg.fit(X_train, y_train)

acc = log_reg.score(X_test, y_test)
print(f"Accuracy: {acc}")

#9.KNN uygula
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#10.Test seti için confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazmak
accuracy_score = accuracy_score(y_test, y_pred)
print(f"accuracy_score: {accuracy_score}")

conf_matris = confusion_matrix(y_test, y_pred)
print(f"confusion matris: \n{conf_matris}")

precision = precision_score(y_test, y_pred)
print(f"Precision: \n{precision}")

recall = recall_score(y_test, y_pred)
print(f"Recall: \n{recall}")

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: \n{f1}")

"""
Bu tür Churn tahminleme görevlerinde genellikle Logistic Regression modeli, KNN algoritmasına kıyasla daha iyi performans gösterir.
Nedenleri:
    1.Logistic Regression her bir özelliğe ayrı ayrı katsayı atar. KNN ise tüm özelliklere eşit mesafe ağırlığı verdiği için gürültü oluşturan değişkenlerden daha fazla etkilenir.
    2.Küçük ve orta ölçekli veri setlerinde doğrusal karar sınırı çizen modeller (Logistic Regression) daha kararlı bir genelleme yaparken, KNN gürültüye karşı daha hassastır.
"""