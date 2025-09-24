# AttnDEC‑KMeans‑HITS (Unified Pipeline)

See usage and structure inside.


## 📦 Datasets (Download & Paths)

Place datasets under `data/` as follows (recommended):

```bash
# InSDN (Kaggle)
kaggle datasets download -d badcodebuilder/insdn-dataset -p data/InSDN --unzip

# UGRansome (Kaggle)
kaggle datasets download -d nkongolo/ugransome-dataset -p data/Ransomware --unzip

# CICIDS2017 (Official site – may require signup)
# Download CSVs then place them under:
# data/CICIDS2017/
```

Then run:
```bash
python experiments/run_insdn.py
python experiments/run_cicids.py
python experiments/run_ransomware.py
```
