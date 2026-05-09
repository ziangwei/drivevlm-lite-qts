from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


class ImageLoader:
    """Load images from normal files, or lazily from a zip archive when files are not extracted."""

    def __init__(self, image_zip: Path | None = None):
        self.image_zip = image_zip
        self._zip: ZipFile | None = ZipFile(image_zip) if image_zip else None
        self._members: dict[str, str] = {}
        self._cache: dict[str, str] = {}
        if self._zip is not None:
            for name in self._zip.namelist():
                if name.endswith("/") or "__MACOSX/" in name:
                    continue
                normalized = self._normalize(name)
                self._members[normalized] = name

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "ImageLoader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def load(self, path: str | Path) -> Image.Image:
        file_path = Path(path)
        if file_path.exists():
            return Image.open(file_path).convert("RGB")
        if self._zip is None:
            raise FileNotFoundError(f"Image not found on disk and no image zip was provided: {path}")

        member = self._resolve_zip_member(path)
        with self._zip.open(member) as handle:
            return Image.open(BytesIO(handle.read())).convert("RGB")

    def load_many(self, paths: list[str | Path]) -> list[Image.Image]:
        return [self.load(path) for path in paths]

    def resolve(self, path: str | Path) -> str:
        file_path = Path(path)
        if file_path.exists():
            return str(file_path)
        if self._zip is None:
            raise FileNotFoundError(f"Image not found on disk and no image zip was provided: {path}")
        return self._resolve_zip_member(path)

    def _resolve_zip_member(self, path: str | Path) -> str:
        normalized = self._normalize(path)
        if normalized in self._cache:
            return self._cache[normalized]

        candidates = self._candidate_keys(normalized)
        for candidate in candidates:
            if candidate in self._members:
                self._cache[normalized] = self._members[candidate]
                return self._members[candidate]

        suffix_matches = [
            member_name
            for member_key, member_name in self._members.items()
            if any(member_key.endswith(candidate) for candidate in candidates)
        ]
        if suffix_matches:
            suffix_matches.sort(key=len)
            self._cache[normalized] = suffix_matches[0]
            return suffix_matches[0]

        raise FileNotFoundError(f"Image not found in zip {self.image_zip}: {path}")

    @staticmethod
    def _normalize(path: str | Path) -> str:
        text = str(path).replace("\\", "/").strip()
        while text.startswith("./"):
            text = text[2:]
        return text.lstrip("/")

    @staticmethod
    def _candidate_keys(normalized: str) -> list[str]:
        candidates = [normalized]
        markers = [
            "data/drivebench/",
            "drivebench/",
            "DriveBench/",
        ]
        lower = normalized.lower()
        for marker in markers:
            marker_lower = marker.lower()
            idx = lower.find(marker_lower)
            if idx >= 0:
                candidates.append(normalized[idx + len(marker) :])
        if "DriveBench/" not in candidates[-1]:
            candidates.append(f"DriveBench/{candidates[-1]}")

        deduped: list[str] = []
        for candidate in candidates:
            candidate = candidate.lstrip("/")
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped
