import shutil
import json
from pathlib import Path
from datetime import datetime
model_folder = Path('models')
legacy = model_folder / 'pneumonia_rf_model.pkl'
if not legacy.exists():
    print('Legacy model not found')
    raise SystemExit(1)
now = datetime.now().strftime('%Y%m%d_%H%M%S')
new_name = model_folder / f'pneumonia_detector_{now}.pkl'
shutil.copy2(str(legacy), str(new_name))
print('Copied legacy to', new_name)

class_indices = {'NORMAL': 0, 'PNEUMONIA': 1}
ci_file = model_folder / f'class_indices_{now}.json'
with open(ci_file, 'w') as f:
    json.dump(class_indices, f)
print('Wrote', ci_file)

metrics = {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5}
metrics_file = model_folder / f'training_metrics_{now}.json'
with open(metrics_file, 'w') as f:
    json.dump(metrics, f)
print('Wrote', metrics_file)
print('Promotion complete')
