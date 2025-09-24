
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay

from .loaders import load_insdn_data, load_cicids_portscan_data, load_ransomware_data
from .attn_autoencoder import build_attn_autoencoder
from .hits_predictor import make_clusters, hits_predict_with_proba
from .metrics import calc_metrics

SEED = 42

def run_pipeline_on_dataset(name, X, y, n_features=10, latent_dim=8, heads=2,
                            kmeans_k_bounds=(8,40), knn_k=12, sim_edge=0.80,
                            epochs=8, batch_size=256, do_plots=True, save_figs=True):
    print(f"\\n\\n================= {name}: Split/Scale/Select =================")
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    k = min(n_features, X_train_sc.shape[1])
    selector = SelectKBest(chi2, k=k)
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_val_sel   = selector.transform(X_val_sc)
    X_test_sel  = selector.transform(X_test_sc)

    sel_cols = np.array(X.columns)[selector.get_support()] if hasattr(X, 'columns') else [f'f{i}' for i in range(k)]
    print(f"[{name}] Selected features (k={k}): {list(sel_cols)}")

    print(f"[{name}] Training AttnDEC autoencoder ...")
    ae, enc = build_attn_autoencoder(input_dim=X_train_sel.shape[1], latent_dim=latent_dim, heads=heads)
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    ae.fit(X_train_sel, X_train_sel,
           validation_data=(X_val_sel, X_val_sel),
           epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2,
           callbacks=[es])

    Z_train = enc.predict(X_train_sel, verbose=0)
    Z_val   = enc.predict(X_val_sel,   verbose=0)
    Z_test  = enc.predict(X_test_sel,  verbose=0)

    print(f"[{name}] Clustering latent space with KMeans ...")
    km, tr_labels, cl_classes, cl_ratio = make_clusters(Z_train, y_train,
                                                        k_min=kmeans_k_bounds[0], k_max=kmeans_k_bounds[1])

    print(f"[{name}] HITS inference on test set ...")
    y_pred, y_proba = [], []
    y_train_arr = np.asarray(y_train).astype(int)
    for z in Z_test:
        lbl, p = hits_predict_with_proba(z, Z_train, y_train_arr, tr_labels, cl_classes, cl_ratio,
                                         kmeans=km, k=knn_k, sim_edge=sim_edge)
        y_pred.append(lbl); y_proba.append(p)

    y_pred = np.array(y_pred, dtype=int)
    y_proba = np.array(y_proba, dtype=float)

    labels_hdr = ['Accuracy','Precision','Recall','F1','ROC-AUC','MCC','Balanced Acc','Specificity','G-mean','PR-AUC','CM']
    m = calc_metrics(np.asarray(y_test).astype(int), y_pred, y_proba)

    print(f"\\n=== Results: AttnDEC-KMeans-HITS — {name} (TEST) ===")
    for kname, val in zip(labels_hdr[:-1], m[:-1]):
        print(f"{kname:>13}: {val:.4f}")
    print("Confusion Matrix (tn, fp, fn, tp):", m[-1])

    if do_plots:
        try:
            fpr, tpr, _ = roc_curve(np.asarray(y_test).astype(int), y_proba)
            fig1 = plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.3f}')
            plt.plot([0,1], [0,1], '--')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'ROC — {name}'); plt.legend(); plt.tight_layout()
            if save_figs:
                fig1.savefig(f"results/figures/roc_{name}.png", dpi=300)
            plt.show()
        except Exception:
            pass

        from sklearn.metrics import precision_recall_curve, average_precision_score
        pre, rec, _ = precision_recall_curve(np.asarray(y_test).astype(int), y_proba)
        ap = average_precision_score(np.asarray(y_test).astype(int), y_proba)
        fig2 = plt.figure(figsize=(6,5))
        plt.plot(rec, pre, label=f'AP={ap:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR — {name}'); plt.legend(); plt.tight_layout()
        if save_figs:
            fig2.savefig(f"results/figures/pr_{name}.png", dpi=300)
        plt.show()

        fig3 = plt.figure()
        ConfusionMatrixDisplay.from_predictions(np.asarray(y_test).astype(int), y_pred)
        plt.title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        if save_figs:
            fig3.savefig(f"results/figures/cm_{name}.png", dpi=300)
        plt.show()

    return dict(dataset=name, metrics=m, features=list(sel_cols))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all", choices=["all","insdn","cicids","ransomware"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    tasks = []
    if args.datasets in ("all",):
        tasks = [("InSDN", load_insdn_data),
                 ("CICIDS-PortScan", load_cicids_portscan_data),
                 ("Ransomware", load_ransomware_data)]
    elif args.datasets == "insdn":
        tasks = [("InSDN", load_insdn_data)]
    elif args.datasets == "cicids":
        tasks = [("CICIDS-PortScan", load_cicids_portscan_data)]
    else:
        tasks = [("Ransomware", load_ransomware_data)]

    all_results = []
    for name, loader in tasks:
        try:
            X, y = loader()
            res = run_pipeline_on_dataset(
                name, X, y,
                n_features=10,
                latent_dim=8,
                heads=2,
                kmeans_k_bounds=(8,40),
                knn_k=12,
                sim_edge=0.80,
                epochs=8,
                batch_size=256,
                do_plots=(not args.no_plots),
                save_figs=True
            )
            all_results.append(res)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if all_results:
        rows = []
        for r in all_results:
            acc, prec, rec, f1, auc_, mcc, bal, spec, gmean, ap, cm = r['metrics']
            rows.append([r['dataset'], acc, prec, rec, f1, auc_, mcc, bal, spec, gmean, ap, cm])
        summary = pd.DataFrame(rows, columns=["Dataset","Acc","Prec","Rec","F1","ROC-AUC","MCC","BalancedAcc","Spec","G-mean","PR-AUC","CM"])
        print("\\n=== Summary ===")
        print(summary.to_string(index=False))
        summary.to_csv("results/metrics/summary.csv", index=False)

if __name__ == "__main__":
    main()
