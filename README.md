# Churn Tahmini Projesi

## Projenin Amacı
Bu projenin amacı, müşteri verilerini (gelir, şehir, üyelik tipi, destek talebi sayısı vb.) analiz ederek müşterilerin hizmeti terk etme (churn) olasılığını tahmin eden makine öğrenmesi modelleri geliştirmektir. 

Proje sürecinde **Pandas** kütüphanesi kullanılarak veri seti yüklenmiş, eksik veriler analiz edilip doldurulmuş, kategorik değişkenler One-Hot Encoding ile dönüştürülmüş ve yeni öznitelikler (feature engineering) türetilmiştir. Ardından scikit-learn ile Logistic Regression ve KNN algoritmalarının performansları karşılaştırılmıştır.

---

## Nasıl Çalıştırılır?

1. **Gereksinimlerin Kurulumu:**
   Pandas ve Scikit-Learn kütüphanelerini yüklemek için terminalde şu komutu çalıştırın:
   ```bash
   pip install pandas scikit-learn
