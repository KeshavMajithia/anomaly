# models/

Place the exported files from Kaggle here after running the export cell.

## Expected files:

| File | Description |
|------|-------------|
| `ztf_transformer.pt` | TransformerAE model weights (~3 MB) |
| `ztf_masked_ae.pt` | MaskedAE model weights (~2 MB) |
| `scaler.pkl` | StandardScaler fitted on 89K ZTF objects |
| `ref_embeddings.npy` | 74K reference embeddings for kNN scoring |
| `class_centroids.json` | Mean embedding per class for auto-triage |
| `config.json` | Model dimensions (N_BINS, N_FEAT, BOTTLENECK, etc.) |

## How to export from Kaggle

Add this cell to your Kaggle notebook and run it, then download `anomaly_hunter_export.zip` from the Output tab:

```python
import joblib, json, shutil, os
import numpy as np

EXPORT = '/kaggle/working/export'
os.makedirs(EXPORT, exist_ok=True)

# Model weights
for f in ['ztf_masked_ae.pt', 'ztf_transformer.pt']:
    shutil.copy(os.path.join(OUT, f), os.path.join(EXPORT, f))

# Scaler
joblib.dump(scaler, os.path.join(EXPORT, 'scaler.pkl'))

# Reference embeddings (used for kNN anomaly scoring)
np.save(os.path.join(EXPORT, 'ref_embeddings.npy'), full_emb2)

# Config
json.dump({
    'N_BINS': N_BINS, 'N_BANDS': N_BANDS,
    'N_FPB': N_FPB, 'N_FEAT': N_FEAT,
    'BOTTLENECK': BOTTLENECK,
    'classes': class_names,
    'n_classes': n_classes,
}, open(os.path.join(EXPORT, 'config.json'), 'w'))

# Class centroids for auto-triage
class_centroids = {}
for cls in unique_classes:
    mask = np.array(lab_classes) == cls
    if mask.sum() > 0:
        class_centroids[cls] = emb2[mask].mean(axis=0).tolist()
json.dump(class_centroids, open(os.path.join(EXPORT, 'class_centroids.json'), 'w'))

# Zip everything
shutil.make_archive('/kaggle/working/anomaly_hunter_export', 'zip', EXPORT)
print("✅ Download anomaly_hunter_export.zip from Output tab")
```
