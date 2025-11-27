import joblib, json, shutil, os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--models-dir", default="models")
parser.add_argument("--archive-dir", default="models/archive")
parser.add_argument("--dry-run", action="store_true", help="Don't move files, just report")
args = parser.parse_args()

models_dir = Path(args.models_dir)
archive_dir = Path(args.archive_dir)
archive_dir.mkdir(parents=True, exist_ok=True)

single_class_models = []

def count_from_class_indices(pkl_path):
    version = pkl_path.stem.replace("pneumonia_detector_", "")
    ci = pkl_path.with_name(f"class_indices_{version}.json")
    if ci.exists():
        try:
            j = json.loads(ci.read_text())
            return len(j.keys()), j
        except Exception:
            return None, None
    return None, None

def count_from_pickle(pkl_path):
    try:
        data = joblib.load(str(pkl_path))
    except Exception:
        return None, None
    
    if isinstance(data, dict):
        # try 'classes' key
        for k in ("classes", "classes_", "class_names"):
            if k in data:
                cls = data[k]
                try:
                    return len(cls), cls
                except Exception:
                    return None, cls
        
        if "model" in data:
            m = data["model"]
            cls = getattr(m, "classes_", None)
            if cls is not None:
                return len(cls), cls
        return None, None
    
    cls = getattr(data, "class_names", None) or getattr(data, "classes_", None) or getattr(data, "classes", None)
    if cls is not None:
        try:
            return len(cls), cls
        except Exception:
            return None, cls
    
    m = getattr(data, "model", None)
    if m is not None:
        cls = getattr(m, "classes_", None)
        if cls is not None:
            return len(cls), cls
    return None, None

for p in sorted(models_dir.glob("pneumonia_detector_*.pkl"), key=os.path.getctime):
    num, src = count_from_class_indices(p)
    if num is None:
        num, src = count_from_pickle(p)
    if num == 1:
        single_class_models.append((p, num, src))
    elif num is None:
        print(f"Could not determine classes for {p.name} (skip or inspect manually).")


legacy = models_dir / "pneumonia_rf_model.pkl"
if legacy.exists():
    num, src = count_from_pickle(legacy)
    if num == 1:
        single_class_models.append((legacy, num, src))

if not single_class_models:
    print("No single-class models detected.")
else:
    print("Single-class model candidates:")
    for p, num, src in single_class_models:
        print(f" - {p.name}   classes={num}   sample={str(src)[:200]}")
    if args.dry_run:
        print("Dry run: no files were moved.")
    else:
        for p, _, _ in single_class_models:
            
            print(f"Archiving {p.name} -> {archive_dir / p.name}")
            shutil.move(str(p), str(archive_dir / p.name))
            
            for pattern in [f"class_indices_{p.stem.replace('pneumonia_detector_', '')}.json",
                            f"training_metrics_{p.stem.replace('pneumonia_detector_', '')}.json"]:
                f = models_dir / pattern
                if f.exists():
                    shutil.move(str(f), str(archive_dir / f.name))
        print("Archived single-class model files to", archive_dir)