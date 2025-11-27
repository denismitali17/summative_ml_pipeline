import joblib
import pprint
from pathlib import Path
p = Path('models') / 'pneumonia_rf_model.pkl'
print('Inspecting:', p)
if not p.exists():
    print('File not found')
    raise SystemExit(1)
obj = joblib.load(str(p))
print('Type:', type(obj))
try:
    keys = list(obj.keys())
    print('Dict keys:', keys)
except Exception:
    attrs = [a for a in dir(obj) if not a.startswith('__')]
    print('Attrs:', attrs[:50])
for candidate in ['classes', 'classes_', 'label_encoder', 'model', 'feature_importances_', 'class_names']:
    if isinstance(obj, dict) and candidate in obj:
        print(candidate, '->', type(obj[candidate]), obj[candidate])
    else:
        v = getattr(obj, candidate, None)
        if v is not None:
            print(candidate, '->', type(v))
            try:
                print('value sample:', getattr(obj, candidate))
            except Exception:
                pass

classes = None
if isinstance(obj, dict):
    for k in ['classes', 'classes_', 'label_encoder', 'classes_']:
        if k in obj:
            classes = obj[k]
            break
if classes is None:
    classes = getattr(obj, 'classes_', None) or getattr(obj, 'classes', None) or getattr(obj, 'label_encoder', None)
print('Resolved classes:', classes)


clf = None
if isinstance(obj, dict):
    clf = obj.get('model')
else:
    clf = getattr(obj, 'model', obj)
print('Classifier type:', type(clf))
print('Has predict_proba?:', hasattr(clf, 'predict_proba'))

pprint.pprint({'classes': classes})
print('Done')
