# 🛡️ ZeroSentinel — Zero-Day Saldırı Tespit Sistemi

XGBoost + Isolation Forest hibrit mimarisi ve SHAP açıklanabilir yapay zeka ile CIC-IDS 2018 veri seti üzerinde zero-day ağ saldırısı tespiti.

---

## 🎯 Proje Özeti

| Özellik | Detay |
|---|---|
| **Veri Seti** | CIC-IDS 2018 (10 CSV, ~11M satır) |
| **Model** | XGBoost + Isolation Forest Hibrit |
| **Açıklanabilirlik** | SHAP (TreeExplainer) |
| **Zero-Day Simülasyonu** | Bot & Infilteration sınıfları eğitimde gizlendi |
| **Arayüz** | Flask Dashboard |

---

## 🗂️ Proje Yapısı

```
zero-day-detection/
├── dashboard/
│   ├── app.py              ← Flask uygulaması
│   └── requirements.txt
├── models/
│   ├── xgboost_hybrid_zeroday.json
│   ├── isolation_forest.pkl
│   ├── final_hybrid_features.json
│   └── model_metadata.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_zero_day_setup.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_shap_analysis.ipynb
│   ├── 05_model_tuning.ipynb
│   └── 06_feature_engineering.ipynb
├── render.yaml
└── .gitignore
```

---

## 🔬 Metodoloji

### Zero-Day Simülasyonu
Modelin hiç görmediği saldırı sınıflarını test setine ekleyerek gerçek zero-day senaryosu oluşturuldu:

| Eğitimde Görülen | Zero-Day (Sadece Testte) |
|---|---|
| DoS, DDoS, SSH BruteForce | **Bot** |
| Web Attack, FTP BruteForce | **Infilteration** |

### Data Leakage Giderimi
SHAP analizi ile `Init Fwd Win Byts` ve `Init Bwd Win Byts` sütunlarının işletim sistemi parmak izini ezberlediği tespit edildi ve modelden çıkarıldı.

### Davranışsal Özellik Mühendisliği
```
Pkt_Size_Ratio      = Fwd Pkt Len Mean / (Bwd Pkt Len Mean + 1)
Duration_per_Packet = Flow Duration / (Toplam Paket + 1)
Fwd_IAT_Ratio       = Fwd IAT Mean / (Flow IAT Mean + 1)
Fwd_Bwd_Pkt_Ratio   = Tot Fwd Pkts / (Tot Bwd Pkts + 1)
```

### Hibrit Mimari
```
Ham Trafik → Feature Engineering → XGBoost ─────┐
                                                   ├→ Hibrit Karar
                  └→ Isolation Forest (IF Score) ──┘
```

---

## 📊 Sonuçlar

* **ROC-AUC:** 0.8637
* **Recall:** %81 (zero-day dahil)
* **Precision:** %46 (saldırılar için)
* **Precision:** %94 (normal trafik için)
* **Accuracy:** %78

