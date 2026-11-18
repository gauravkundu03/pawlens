import os
import shutil
import hashlib
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")


def inspect_image(args):
    # Returns one metadata record; never raises (errors are recorded, not fatal)
    path, label = args
    rec = dict(path=path, label=label, ext=os.path.splitext(path)[1].lower(),
               file_size=os.path.getsize(path), width=np.nan, height=np.nan,
               channels=np.nan, mode=None, format=None, valid=False, error="", md5=None)
    try:
        if rec["file_size"] == 0:
            raise ValueError("empty file (0 bytes)")
        with Image.open(path) as im:
            rec.update(width=im.width, height=im.height, mode=im.mode,
                       format=im.format, channels=len(im.getbands()))
            im.verify()
        with Image.open(path) as im:   # verify() does not decode pixels; load() does
            im.load()
        rec["valid"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        with open(path, "rb") as f:
            rec["md5"] = hashlib.md5(f.read()).hexdigest()
    except Exception:
        pass
    return rec


def scan_dataset(root, class_names=("Cat", "Dog"), workers=16):
    # Recursively scan root/<class>/** and build a metadata DataFrame
    items = []
    for cls in class_names:
        for dp, _, files in os.walk(os.path.join(root, cls)):
            for fn in files:
                items.append((os.path.join(dp, fn), cls))
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(inspect_image, items))
    return pd.DataFrame(recs)


def quarantine_invalid(df, log_csv, remove=False):
    # Record every invalid file in a CSV log; optionally delete them from disk
    bad = df[~df["valid"]].copy()
    bad.to_csv(log_csv, index=False)
    if remove:
        for p in bad["path"]:
            try:
                os.remove(p)
            except OSError:
                pass
    return bad


def build_cleaned_dataset(df, out_dir, min_side=32):
    # Drop invalid, conflicting-label duplicates, exact duplicates and tiny images; copy the rest
    steps = {"initial files": len(df)}
    d = df[df["valid"]].copy()
    steps["after removing corrupted / unreadable"] = len(d)
    nl = d.groupby("md5")["label"].nunique()
    conflict = set(nl[nl > 1].index)
    d = d[~d["md5"].isin(conflict)]
    steps["after removing cross-class duplicates"] = len(d)
    d = d.drop_duplicates("md5", keep="first")
    steps["after removing exact duplicates"] = len(d)
    d = d[(d["width"] >= min_side) & (d["height"] >= min_side)]
    steps[f"after removing images with side < {min_side}px"] = len(d)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    new_paths = []
    for i, (p, lab, ext) in enumerate(zip(d["path"], d["label"], d["ext"])):
        dst_dir = os.path.join(out_dir, lab)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{lab.lower()}_{i:05d}{ext}")
        shutil.copy2(p, dst)
        new_paths.append(dst)
    d = d.copy()
    d["orig_path"] = d["path"]
    d["path"] = new_paths
    return d.reset_index(drop=True), pd.Series(steps, name="images")
