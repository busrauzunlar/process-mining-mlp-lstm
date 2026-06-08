import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.preprocessing import (
    load_and_clean_data,
    extract_session_features,
    extract_nlp_features,
    generate_labels,
    prepare_mlp_data,
    prepare_sequence_data
)
from src.models import (
    create_mlp_model,
    create_lstm_model,
    create_gru_model
)
from evaluation import (
    evaluate_model,
    plot_cm,
    plot_roc_curves,
    plot_pr_curves,
    plot_training_history,
    compare_models
)

# File Paths
FILE1 = r"C:\Users\BÜŞRA\Desktop\cyber_nn_project\data\data1\process_mining.csv"
FILE2 = r"C:\Users\BÜŞRA\Desktop\cyber_nn_project\data\data2\process_mining.csv"

def main():
    print("=== Süreç Madenciliği Anomali Tespiti Projesi Başlatılıyor ===")
    
    # 1. Veri Yükleme ve Temizleme
    print("\n[1/5] Veri yükleniyor ve temizleniyor...")
    df = load_and_clean_data(FILE1, FILE2)
    print(f"Toplam Log Satırı: {len(df)}")
    print(f"Toplam Oturum (Sandbox) Sayısı: {df['sandbox_id'].nunique()}")
    
    # 2. Özellik Mühendisliği (Tablosal ve NLP)
    print("\n[2/5] Özellik mühendisliği uygulanıyor...")
    session_features = extract_session_features(df)
    tfidf_features = extract_nlp_features(df, max_features=20)
    print(f"Tablosal özellikler boyutu: {session_features.shape}")
    print(f"TF-IDF NLP özellikleri boyutu: {tfidf_features.shape}")
    
    # 3. Anomali Etiketleme Karşılaştırması
    print("\n[3/5] Anomali etiketleme stratejileri değerlendiriliyor...")
    labels_pct = generate_labels(session_features, method="percentile_95")
    labels_iforest = generate_labels(session_features, method="isolation_forest", contamination=0.10)
    labels_lof = generate_labels(session_features, method="local_outlier_factor", contamination=0.10)
    
    print(f"  - 95. Persentil Yöntemi Anomali Sayısı: {sum(labels_pct)} / {len(labels_pct)}")
    print(f"  - Isolation Forest Yöntemi Anomali Sayısı: {sum(labels_iforest)} / {len(labels_iforest)}")
    print(f"  - Local Outlier Factor Yöntemi Anomali Sayısı: {sum(labels_lof)} / {len(labels_lof)}")
    
    # Karşılaştırma Analizi için kesişimler
    iforest_pct_overlap = sum((labels_iforest == 1) & (labels_pct == 1))
    lof_pct_overlap = sum((labels_lof == 1) & (labels_pct == 1))
    iforest_lof_overlap = sum((labels_iforest == 1) & (labels_lof == 1))
    print(f"  - Isolation Forest & 95. Persentil Kesişimi: {iforest_pct_overlap}")
    print(f"  - LOF & 95. Persentil Kesişimi: {lof_pct_overlap}")
    print(f"  - Isolation Forest & LOF Kesişimi: {iforest_lof_overlap}")
    
    # Ana etiketleme yöntemi olarak Isolation Forest seçiliyor
    chosen_labels = labels_iforest
    session_features["label"] = chosen_labels
    
    # 4. Modellerin Eğitilmesi ve Test Edilmesi
    print("\n[4/5] Modeller eğitiliyor (Etiketleme: Isolation Forest)...")
    results = []
    
    # --- A. MLP Model Varyasyonları ---
    # MLP veri setini birleştir
    full_df = pd.merge(session_features, tfidf_features, on="sandbox_id")
    X_mlp = full_df.drop(columns=["sandbox_id", "label"])
    y_mlp = full_df["label"].values
    
    # MLP 1: No Class Balancing
    print("\n>>> MLP (Dengesiz/Orijinal) eğitiliyor...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_mlp, y_mlp, test_size=0.2, stratify=y_mlp, random_state=42
    )
    # Scale raw data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_raw_s = scaler.fit_transform(X_train_raw)
    X_test_raw_s = scaler.transform(X_test_raw)
    
    mlp_raw = create_mlp_model(X_train_raw_s.shape[1])
    history_mlp_raw = mlp_raw.fit(
        X_train_raw_s, y_train_raw,
        epochs=40, batch_size=16,
        validation_data=(X_test_raw_s, y_test_raw),
        verbose=0
    )
    res_mlp_raw = evaluate_model(mlp_raw, X_test_raw_s, y_test_raw, "MLP (Orijinal)")
    res_mlp_raw["y_true"] = y_test_raw
    results.append(res_mlp_raw)
    plot_training_history(history_mlp_raw, "MLP (Orijinal)", "mlp_original")
    plot_cm(y_test_raw, res_mlp_raw["y_pred"], "MLP (Orijinal) Confusion Matrix", "mlp_original_cm.png")

    # MLP 2: SMOTE ile Dengelenmiş
    print("\n>>> MLP (SMOTE ile Dengelenmiş) eğitiliyor...")
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = prepare_mlp_data(
        session_features, tfidf_features, chosen_labels, balance_method="smote"
    )
    mlp_sm = create_mlp_model(X_train_sm.shape[1])
    history_mlp_sm = mlp_sm.fit(
        X_train_sm, y_train_sm,
        epochs=40, batch_size=16,
        validation_data=(X_test_sm, y_test_sm),
        verbose=0
    )
    res_mlp_sm = evaluate_model(mlp_sm, X_test_sm, y_test_sm, "MLP (SMOTE)")
    res_mlp_sm["y_true"] = y_test_sm
    results.append(res_mlp_sm)
    plot_training_history(history_mlp_sm, "MLP (SMOTE)", "mlp_smote")
    plot_cm(y_test_sm, res_mlp_sm["y_pred"], "MLP (SMOTE) Confusion Matrix", "mlp_smote_cm.png")
    
    # MLP 3: RandomOverSampler ile Dengelenmiş
    print("\n>>> MLP (ROS ile Dengelenmiş) eğitiliyor...")
    X_train_ros, X_test_ros, y_train_ros, y_test_ros = prepare_mlp_data(
        session_features, tfidf_features, chosen_labels, balance_method="ros"
    )
    mlp_ros = create_mlp_model(X_train_ros.shape[1])
    history_mlp_ros = mlp_ros.fit(
        X_train_ros, y_train_ros,
        epochs=40, batch_size=16,
        validation_data=(X_test_ros, y_test_ros),
        verbose=0
    )
    res_mlp_ros = evaluate_model(mlp_ros, X_test_ros, y_test_ros, "MLP (ROS)")
    res_mlp_ros["y_true"] = y_test_ros
    results.append(res_mlp_ros)
    plot_training_history(history_mlp_ros, "MLP (ROS)", "mlp_ros")
    plot_cm(y_test_ros, res_mlp_ros["y_pred"], "MLP (ROS) Confusion Matrix", "mlp_ros_cm.png")

    # --- B. Sıralı Modeller (LSTM & GRU) ---
    print("\n>>> Sıralı veri seti hazırlanıyor...")
    X_seq, vocab_size = prepare_sequence_data(df, chosen_labels, max_len=50)
    print(f"Sıralı Veri Boyutu: {X_seq.shape}, Sözlük (Vocab) Boyutu: {vocab_size}")
    
    # Stratified Split (Problem 2 Çözümü)
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, chosen_labels, test_size=0.2, stratify=chosen_labels, random_state=42
    )
    
    # LSTM Modeli
    print("\n>>> Çift Yönlü LSTM (Bidirectional LSTM) eğitiliyor...")
    lstm_model = create_lstm_model(vocab_size=vocab_size, max_len=50)
    history_lstm = lstm_model.fit(
        X_seq_train, y_seq_train,
        epochs=30, batch_size=16,
        validation_data=(X_seq_test, y_seq_test),
        verbose=0
    )
    res_lstm = evaluate_model(lstm_model, X_seq_test, y_seq_test, "LSTM (Bidirectional)")
    res_lstm["y_true"] = y_seq_test
    results.append(res_lstm)
    plot_training_history(history_lstm, "LSTM (Bidirectional)", "lstm_bidirectional")
    plot_cm(y_seq_test, res_lstm["y_pred"], "LSTM (Bidirectional) Confusion Matrix", "lstm_cm.png")
    
    # GRU Modeli (Yeni Karşılaştırma Modeli)
    print("\n>>> Çift Yönlü GRU (Bidirectional GRU) eğitiliyor...")
    gru_model = create_gru_model(vocab_size=vocab_size, max_len=50)
    history_gru = gru_model.fit(
        X_seq_train, y_seq_train,
        epochs=30, batch_size=16,
        validation_data=(X_seq_test, y_seq_test),
        verbose=0
    )
    res_gru = evaluate_model(gru_model, X_seq_test, y_seq_test, "GRU (Bidirectional)")
    res_gru["y_true"] = y_seq_test
    results.append(res_gru)
    plot_training_history(history_gru, "GRU (Bidirectional)", "gru_bidirectional")
    plot_cm(y_seq_test, res_gru["y_pred"], "GRU (Bidirectional) Confusion Matrix", "gru_cm.png")
    
    # 5. Model Karşılaştırma ve Raporlama
    print("\n[5/5] Sonuçlar karşılaştırılıyor ve grafikler kaydediliyor...")
    plot_roc_curves(results, filename="roc_curves.png")
    plot_pr_curves(results, filename="pr_curves.png")
    compare_models(results, filename="model_comparison.png")
    
    # Ekrana Özet Tablo Basma
    summary_df = pd.DataFrame([
        {
            "Model": r["name"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "Specificity": f"{r['specificity']:.4f}",
            "F1-Score": f"{r['f1']:.4f}",
            "ROC-AUC": f"{r['roc_auc']:.4f}"
        }
        for r in results
    ])
    
    print("\n" + "="*80)
    print("                      MODEL PERFORMANS KARŞILAŞTIRMASI")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    print("\nTüm grafikler ve karşılaştırma sonuçları 'outputs/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()