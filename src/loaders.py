
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42

def load_insdn_data(base_dir="data/InSDN"):
    files = {
        "metasploitable-2.csv": 1,
        "Normal_data.csv": 0,
        "OVS.csv": 0
    }
    dfs = []
    for fname, label in files.items():
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing InSDN file: {path}")
        df = pd.read_csv(path, low_memory=False)

        drop_substr = ['flow id', 'src ip', 'src port', 'dst ip', 'dst port',
                       'timestamp', 'unnamed', 'fwd header length.1']
        cols_to_drop = [c for c in df.columns if any(s in c.lower() for s in drop_substr)]
        cols_to_drop += [c for c in df.columns if c.lower() in {'id', 'pid'}]
        df = df.drop(columns=list(set(cols_to_drop)), errors='ignore')

        df['Label'] = label
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    data_num = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = data_num.drop(columns=['Label'], errors='ignore')
    y = data_num['Label'].astype(int)
    print(f"[InSDN] X={X.shape}, positives={int((y==1).sum())}, negatives={int((y==0).sum())}")
    return X, y

def load_cicids_portscan_data(path="data/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CICIDS PortScan file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if 'Label' not in df.columns:
        raise KeyError("CICIDS file must contain column 'Label'")
    df['Label'] = df['Label'].astype(str).apply(lambda x: 1 if 'portscan' in x.lower() else 0)

    drop_candidates = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors='ignore')

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    data_num = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = data_num.drop(columns=['Label'], errors='ignore')
    y = data_num['Label'].astype(int)
    print(f"[CICIDS-PortScan] X={X.shape}, positives={int((y==1).sum())}, negatives={int((y==0).sum())}")
    return X, y

def load_ransomware_data(path="data/Ransomware/final(2).csv"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing Ransomware file: {path}")
    df = pd.read_csv(path, low_memory=False)

    drop_np = ['Time', 'SeddAddress', 'ExpAddress', 'IPaddress']
    df = df.drop(columns=[c for c in drop_np if c in df.columns], errors='ignore')

    rename_map = {'Protcol': 'Protocol'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    keep = ['Protocol', 'Flag', 'Family', 'Netflow_Bytes', 'BTC', 'USD', 'Threats', 'Port', 'Prediction']
    exists = [c for c in keep if c in df.columns]
    df = df[exists]

    if 'Prediction' not in df.columns:
        raise KeyError("Ransomware file must contain column 'Prediction'")

    def map_binary(v):
        s = str(v).strip().lower()
        if s in {'benign','normal','0','none','clean'}:
            return 0
        return 1
    y = df['Prediction'].apply(map_binary).astype(int)

    cat_cols = [c for c in ['Protocol', 'Flag', 'Family', 'Threats'] if c in df.columns]
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    if 'Prediction' in df.columns:
        df = df.drop(columns=['Prediction'], errors='ignore')
    X = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    print(f"[Ransomware] X={X.shape}, positives={int((y==1).sum())}, negatives={int((y==0).sum())}")
    return X, y
