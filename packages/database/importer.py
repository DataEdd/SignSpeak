"""Import signs from external sources (WLASL, How2Sign)."""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import Sign, SignStatus, VideoInfo


class SignImporter(ABC):
    """Base class for sign importers."""

    def __init__(self, source_path: str | Path, store: "SignStore"):
        """Initialize importer.

        Args:
            source_path: Path to source dataset
            store: SignStore instance for importing into
        """
        self.source_path = Path(source_path)
        self.store = store

    @abstractmethod
    def import_all(self) -> tuple[int, list[str]]:
        """Import all signs from source.

        Returns:
            Tuple of (success_count, list of error messages)
        """
        pass

    @abstractmethod
    def import_sign(self, gloss: str) -> Optional[Sign]:
        """Import a single sign by gloss.

        Args:
            gloss: The sign gloss to import

        Returns:
            Imported Sign or None if not found
        """
        pass


class WLASLImporter(SignImporter):
    """Import signs from WLASL dataset.

    WLASL structure:
    wlasl/
    ├── WLASL_v0.3.json  # Metadata file
    └── videos/
        ├── 00001.mp4
        ├── 00002.mp4
        └── ...
    """

    def __init__(
        self,
        source_path: str | Path,
        store: "SignStore",
        metadata_file: str = "WLASL_v0.3.json",
    ):
        super().__init__(source_path, store)
        self.metadata_file = metadata_file
        self._metadata: Optional[list] = None

    def _load_metadata(self) -> list:
        """Load WLASL metadata JSON."""
        if self._metadata is None:
            meta_path = self.source_path / self.metadata_file
            if not meta_path.exists():
                raise FileNotFoundError(f"WLASL metadata not found: {meta_path}")
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)
        return self._metadata

    def _find_video(self, video_id: str) -> Optional[Path]:
        """Find video file by ID."""
        videos_dir = self.source_path / "videos"
        for ext in [".mp4", ".webm", ".avi"]:
            video_path = videos_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
        return None

    def import_all(self, limit: Optional[int] = None) -> tuple[int, list[str]]:
        """Import all signs from WLASL.

        Args:
            limit: Optional limit on number of signs to import

        Returns:
            Tuple of (success_count, list of error messages)
        """
        metadata = self._load_metadata()
        success = 0
        errors = []

        for i, entry in enumerate(metadata):
            if limit and success >= limit:
                break

            gloss = entry.get("gloss", "").upper()
            if not gloss:
                continue

            try:
                sign = self._import_entry(entry)
                if sign:
                    success += 1
            except Exception as e:
                errors.append(f"{gloss}: {str(e)}")

        return success, errors

    def _import_entry(self, entry: dict) -> Optional[Sign]:
        """Import a single WLASL entry."""
        gloss = entry.get("gloss", "").upper()
        instances = entry.get("instances", [])

        if not instances:
            return None

        # Use the first instance
        instance = instances[0]
        video_id = instance.get("video_id", "")

        video_path = self._find_video(video_id)
        if not video_path:
            return None

        # Check if already exists
        existing = self.store.get_sign(gloss)
        if existing:
            return None

        # Create destination directory
        dest_dir = self.store.base_path / "imported" / "wlasl" / gloss
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy video
        video_dest = dest_dir / "video.mp4"
        shutil.copy2(video_path, video_dest)

        # Create sign metadata
        sign = Sign(
            gloss=gloss,
            english=[gloss.lower().replace("_", " ")],
            source="wlasl",
            status=SignStatus.IMPORTED,
            video=VideoInfo(
                file="video.mp4",
                fps=instance.get("fps", 25),
            ),
            path=dest_dir,
        )

        # Save metadata
        metadata_path = dest_dir / "sign.json"
        with open(metadata_path, "w") as f:
            json.dump(sign.to_dict(), f, indent=2)

        return sign

    def import_sign(self, gloss: str) -> Optional[Sign]:
        """Import a specific sign by gloss."""
        gloss = gloss.upper()
        metadata = self._load_metadata()

        for entry in metadata:
            if entry.get("gloss", "").upper() == gloss:
                return self._import_entry(entry)

        return None

    def list_available(self) -> list[str]:
        """List all glosses available in WLASL."""
        metadata = self._load_metadata()
        return [entry.get("gloss", "").upper() for entry in metadata if entry.get("gloss")]


class How2SignImporter(SignImporter):
    """Import signs from How2Sign dataset.

    How2Sign structure:
    how2sign/
    ├── train/
    │   ├── clips/
    │   │   └── *.mp4
    │   └── annotations.csv
    └── test/
        └── ...
    """

    def __init__(
        self,
        source_path: str | Path,
        store: "SignStore",
        split: str = "train",
    ):
        super().__init__(source_path, store)
        self.split = split
        self._annotations: Optional[list] = None

    def _load_annotations(self) -> list:
        """Load annotations CSV."""
        if self._annotations is not None:
            return self._annotations

        import csv

        annotations_path = self.source_path / self.split / "annotations.csv"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")

        self._annotations = []
        with open(annotations_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._annotations.append(row)

        return self._annotations

    def import_all(self, limit: Optional[int] = None) -> tuple[int, list[str]]:
        """Import signs from How2Sign.

        Args:
            limit: Optional limit on number of signs

        Returns:
            Tuple of (success_count, error messages)
        """
        annotations = self._load_annotations()
        success = 0
        errors = []
        seen_glosses = set()

        for row in annotations:
            if limit and success >= limit:
                break

            gloss = row.get("gloss", "").upper()
            if not gloss or gloss in seen_glosses:
                continue
            seen_glosses.add(gloss)

            try:
                sign = self._import_row(row)
                if sign:
                    success += 1
            except Exception as e:
                errors.append(f"{gloss}: {str(e)}")

        return success, errors

    def _import_row(self, row: dict) -> Optional[Sign]:
        """Import a single annotation row."""
        gloss = row.get("gloss", "").upper()
        video_file = row.get("video_file", "")

        video_path = self.source_path / self.split / "clips" / video_file
        if not video_path.exists():
            return None

        existing = self.store.get_sign(gloss)
        if existing:
            return None

        # Create destination
        dest_dir = self.store.base_path / "imported" / "how2sign" / gloss
        dest_dir.mkdir(parents=True, exist_ok=True)

        video_dest = dest_dir / "video.mp4"
        shutil.copy2(video_path, video_dest)

        sign = Sign(
            gloss=gloss,
            english=[gloss.lower().replace("_", " ")],
            source="how2sign",
            status=SignStatus.IMPORTED,
            video=VideoInfo(file="video.mp4"),
            path=dest_dir,
        )

        metadata_path = dest_dir / "sign.json"
        with open(metadata_path, "w") as f:
            json.dump(sign.to_dict(), f, indent=2)

        return sign

    def import_sign(self, gloss: str) -> Optional[Sign]:
        """Import a specific sign by gloss."""
        gloss = gloss.upper()
        annotations = self._load_annotations()

        for row in annotations:
            if row.get("gloss", "").upper() == gloss:
                return self._import_row(row)

        return None


def create_importer(
    source_type: str,
    source_path: str | Path,
    store: "SignStore",
    **kwargs,
) -> SignImporter:
    """Factory function to create appropriate importer.

    Args:
        source_type: Type of source ("wlasl", "how2sign")
        source_path: Path to source data
        store: SignStore instance
        **kwargs: Additional args for specific importer

    Returns:
        Appropriate SignImporter instance
    """
    importers = {
        "wlasl": WLASLImporter,
        "how2sign": How2SignImporter,
    }

    if source_type not in importers:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(importers.keys())}")

    return importers[source_type](source_path, store, **kwargs)
