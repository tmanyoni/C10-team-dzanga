"""Download the competition data using the Kaggle API."""
import subprocess, sys, zipfile
from pathlib import Path

COMPETITION = "agricultural-extension-rag-smart-retrieval-for-farmers"
RAW = Path(__file__).parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading competition data...")
    subprocess.check_call([sys.executable, "-m", "kaggle",
                           "competitions", "download",
                           "-c", COMPETITION, "-p", str(RAW)])
    for z in RAW.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        z.unlink()
    print("Done. Files in", RAW)

if __name__ == "__main__":
    main()