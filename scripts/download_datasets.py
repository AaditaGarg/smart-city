"""Download and unpack public datasets into data/raw/.

  python scripts/download_datasets.py kitti                   # vehicles (~390 MB, auto)
  python scripts/download_datasets.py rdd2022 --countries India Czech
  python scripts/download_datasets.py rdd2022 --local-zip ~/Downloads/RDD2022_India.zip
  python scripts/download_datasets.py stats                   # real counts of what is on disk

Every run appends to data/raw/MANIFEST.json (source, license, file size, sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.json"

KITTI_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip"
FIGSHARE_API = "https://api.figshare.com/v2/articles/21431547"  # RDD2022 record
RDD_FILE_NAMES = {  # per-country archive names listed on the RDD2022 GitHub page
    "Japan": "RDD2022_Japan.zip", "India": "RDD2022_India.zip", "Czech": "RDD2022_Czech.zip",
    "Norway": "RDD2022_Norway.zip", "United_States": "RDD2022_United_States.zip",
    "China_MotorBike": "RDD2022_China_MotorBike.zip", "China_Drone": "RDD2022_China_Drone.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def record(name: str, source: str, license_: str, archive: Path) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    data[name] = {"source": source, "license": license_, "archive": archive.name,
                  "bytes": archive.stat().st_size, "sha256": sha256(archive),
                  "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    MANIFEST.write_text(json.dumps(data, indent=2))


def unzip(archive: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        print(f"[skip] {out} already extracted")
        return
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)


def get_kitti(local_zip: str | None) -> None:
    archive = Path(local_zip).expanduser() if local_zip else download(KITTI_URL, RAW / "_archives" / "kitti.zip")
    unzip(archive, RAW / "kitti")
    record("kitti", KITTI_URL, "KITTI terms: non-commercial research use (cvlibs.net/datasets/kitti)", archive)


def figshare_urls() -> dict[str, str]:
    r = requests.get(FIGSHARE_API, timeout=30)
    r.raise_for_status()
    return {f["name"]: f["download_url"] for f in r.json().get("files", [])}


def get_rdd(countries: list[str], local_zip: str | None) -> None:
    jobs: list[tuple[str, Path]] = []
    if local_zip:
        zp = Path(local_zip).expanduser()
        jobs.append((zp.stem.replace("RDD2022_", ""), zp))
    else:
        try:
            urls = figshare_urls()
        except Exception as e:
            sys.exit(f"Could not query Figshare ({e}).\nDownload the zips manually from "
                     "https://github.com/sekilab/RoadDamageDetector (links per country) and re-run with --local-zip.")
        print("Files listed on Figshare:", ", ".join(urls) or "none")
        for c in countries:
            name = RDD_FILE_NAMES.get(c)
            url = urls.get(name) if name else None
            if not url and "RDD2022.zip" in urls:
                print(f"[info] {name} not on Figshare; the combined RDD2022.zip is available (large). Re-run with --countries all-in-one is not automated; use --local-zip.")
            if not url:
                print(f"[warn] {c}: no direct file found. Download manually and use --local-zip.")
                continue
            jobs.append((c, download(url, RAW / "_archives" / name)))
    for country, archive in jobs:
        unzip(archive, RAW / "rdd2022" / country)
        record(f"rdd2022_{country}", "https://figshare.com/articles/dataset/21431547", "CC BY-SA 4.0", archive)


def stats() -> None:
    k = RAW / "kitti"
    if k.exists():
        names = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
        for s in ("train", "val"):
            imgs = list((k / "images" / s).glob("*.png")) if (k / "images" / s).exists() else []
            cnt = Counter()
            for lf in (k / "labels" / s).glob("*.txt"):
                for line in lf.read_text().splitlines():
                    if line.strip():
                        cnt[names[int(line.split()[0])]] += 1
            print(f"KITTI {s}: {len(imgs)} images, boxes: {dict(cnt)}")
    r = RAW / "rdd2022"
    if r.exists():
        import xml.etree.ElementTree as ET
        for cdir in sorted(p for p in r.iterdir() if p.is_dir()):
            xmls = list(cdir.rglob("*.xml"))
            imgs = list(cdir.rglob("*.jpg"))
            cnt = Counter()
            for x in xmls:
                for o in ET.parse(x).getroot().iter("object"):
                    cnt[o.findtext("name")] += 1
            print(f"RDD2022 {cdir.name}: {len(imgs)} jpgs, {len(xmls)} xml annotations, boxes: {dict(cnt)}")
    if not k.exists() and not r.exists():
        print("Nothing in data/raw yet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("kitti"); pk.add_argument("--local-zip")
    pr = sub.add_parser("rdd2022"); pr.add_argument("--countries", nargs="+", default=["India"], choices=list(RDD_FILE_NAMES))
    pr.add_argument("--local-zip")
    sub.add_parser("stats")
    a = ap.parse_args()
    {"kitti": lambda: get_kitti(a.local_zip), "rdd2022": lambda: get_rdd(a.countries, a.local_zip), "stats": stats}[a.cmd]()
