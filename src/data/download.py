"""Download and organize the Cats vs Dogs dataset."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from PIL import Image
from tqdm import tqdm

from src import config


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_zip(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing archive: {dest}")
        return dest

    print(f"Downloading {url}")
    with urlopen(url, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024
        with open(dest, "wb") as out, tqdm(
            total=total or None, unit="B", unit_scale=True, desc="download"
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    return dest


def extract_and_organize(zip_path: Path, raw_dir: Path) -> dict:
    """Extract zip and place images under raw/PetImages/{Cat,Dog}/."""
    extract_root = raw_dir / "_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # Archive layout: PetImages/Cat, PetImages/Dog
    pet_src = extract_root / "PetImages"
    if not pet_src.exists():
        # Fallback: search for Cat/Dog folders
        candidates = list(extract_root.rglob("Cat"))
        if not candidates:
            raise FileNotFoundError("Could not find PetImages/Cat in archive")
        pet_src = candidates[0].parent

    target = raw_dir / "PetImages"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(pet_src, target)
    shutil.rmtree(extract_root)

    counts = {"cat": 0, "dog": 0, "corrupt": 0}
    for class_name, folder in (("cat", "Cat"), ("dog", "Dog")):
        class_dir = target / folder
        for img_path in list(class_dir.glob("*")):
            if not img_path.is_file():
                continue
            if not _is_valid_image(img_path):
                img_path.unlink(missing_ok=True)
                counts["corrupt"] += 1
                continue
            counts[class_name] += 1

    return counts


def create_synthetic_dataset(
    raw_dir: Path,
    n_per_class: int = 64,
    image_size: int = 64,
) -> dict:
    """Create tiny synthetic RGB images for CI / local smoke runs."""
    target = raw_dir / "PetImages"
    if target.exists():
        shutil.rmtree(target)

    counts = {"cat": 0, "dog": 0, "corrupt": 0}
    for class_name, folder, color in (
        ("cat", "Cat", (220, 120, 80)),
        ("dog", "Dog", (80, 140, 220)),
    ):
        class_dir = target / folder
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            pixels = bytes(
                [
                    (color[0] + (i * 3) % 30) % 256,
                    (color[1] + (i * 5) % 30) % 256,
                    (color[2] + (i * 7) % 30) % 256,
                ]
                * (image_size * image_size)
            )
            img = Image.frombytes("RGB", (image_size, image_size), pixels)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            out = class_dir / f"{class_name}_{i:04d}.jpg"
            out.write_bytes(buf.getvalue())
            counts[class_name] += 1
    return counts


def _count_existing_images(raw_dir: Path) -> dict[str, int] | None:
    """Return class counts if PetImages looks already prepared; else None."""
    pet = raw_dir / "PetImages"
    cat_dir = pet / "Cat"
    dog_dir = pet / "Dog"
    if not cat_dir.is_dir() or not dog_dir.is_dir():
        return None

    def _count(folder: Path) -> int:
        return sum(
            1
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

    counts = {"cat": _count(cat_dir), "dog": _count(dog_dir), "corrupt": 0}
    # Real dataset has ~12.5k/class; require a sensible minimum to treat as present
    if counts["cat"] >= 100 and counts["dog"] >= 100:
        return counts
    return None


def ensure_dataset(
    synthetic: bool = False,
    n_per_class: int = 64,
    force: bool = False,
) -> dict:
    raw_dir = config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        counts = create_synthetic_dataset(raw_dir, n_per_class=n_per_class)
        print(f"Synthetic dataset ready: {counts}")
        return counts

    if not force:
        existing = _count_existing_images(raw_dir)
        if existing is not None:
            print(
                f"Dataset already present under {raw_dir / 'PetImages'}: {existing} "
                "(use --force to re-download)"
            )
            return existing

    zip_path = raw_dir / config.DATASET_ZIP_NAME
    download_zip(config.DATASET_URL, zip_path)
    counts = extract_and_organize(zip_path, raw_dir)
    print(f"Dataset ready under {raw_dir / 'PetImages'}: {counts}")
    return counts
