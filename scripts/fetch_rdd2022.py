"""Fetch ONLY one country (default: India) from the RDD2022 combined archive.

Online (reads just the needed bytes of the big zip via HTTP range requests):
    python scripts/fetch_rdd2022.py --country India
Offline (you downloaded RDD2022_released_through_CRDDC2022.zip in a browser):
    python scripts/fetch_rdd2022.py --country India --zip ~/Downloads/RDD2022_released_through_CRDDC2022.zip

Only labelled `train` files (images + XML) are extracted; the `test` split has no labels.
Output: data/raw/rdd2022/<Country>/...   Then run: python scripts/download_datasets.py stats
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "rdd2022"
FIGSHARE_API = "https://api.figshare.com/v2/articles/21431547"
COMBINED_NAME = "RDD2022_released_through_CRDDC2022.zip"


def wanted(name: str, country: str, nested: bool = False) -> bool:
    parts = [p.lower() for p in name.split("/") if p]
    return (not name.endswith("/") and (nested or country.lower() in parts)
            and "test" not in parts and "__macosx" not in parts
            and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".xml"})


def dest_for(name: str, country: str, nested: bool = False) -> Path:
    parts = [p for p in name.split("/") if p]
    lower = [p.lower() for p in parts]
    idx = lower.index(country.lower()) if country.lower() in lower else -1
    return OUT / country / Path(*parts[idx + 1:])


def extract(z: zipfile.ZipFile, country: str, nested: bool = False) -> int:
    members = [i for i in z.infolist() if wanted(i.filename, country, nested)]
    if not members:
        sample = [i.filename for i in z.infolist()[:15]]
        sys.exit(f"No files matched country '{country}'. First entries in the archive: {sample}")
    for info in tqdm(members, desc=f"extract {country}", unit="file"):
        d = dest_for(info.filename, country, nested)
        if d.exists() and d.stat().st_size == info.file_size:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(d, "wb") as dst:
            dst.write(src.read())
    return len(members)


def nested_zip_member(z: zipfile.ZipFile, country: str):
    """The combined archive stores one zip per country, e.g. RDD2022/India.zip."""
    for i in z.infolist():
        if Path(i.filename).name.lower() == f"{country.lower()}.zip":
            return i
    return None


def extract_country(z: zipfile.ZipFile, country: str) -> int:
    member = nested_zip_member(z, country)
    if member is None:
        return extract(z, country)  # archive stores plain folders instead
    tmp = ROOT / "data" / "raw" / "_archives" / f"RDD2022_{country}.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not (tmp.exists() and tmp.stat().st_size == member.file_size):
        with z.open(member) as src, open(tmp, "wb") as dst, tqdm(
                total=member.file_size, unit="B", unit_scale=True, desc=f"download {country}.zip") as bar:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(tmp) as inner:
        return extract(inner, country, nested=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="India")
    ap.add_argument("--zip", help="local path to the combined RDD2022 zip")
    a = ap.parse_args()

    if a.zip:
        with zipfile.ZipFile(Path(a.zip).expanduser()) as z:
            n = extract_country(z, a.country)
    else:
        try:
            from remotezip import RemoteZip
        except ImportError:
            sys.exit("Run: pip install remotezip   (or download the zip in a browser and pass --zip)")
        files = {f["name"]: f["download_url"] for f in requests.get(FIGSHARE_API, timeout=30).json()["files"]}
        url = files.get(COMBINED_NAME)
        if not url:
            sys.exit(f"{COMBINED_NAME} not listed on Figshare. Found: {list(files)}")
        print("Reading remote archive index (no full download)...")
        with RemoteZip(url) as z:
            n = extract_country(z, a.country)
    print(f"Done: {n} files for {a.country} in {OUT / a.country}")


if __name__ == "__main__":
    main()
