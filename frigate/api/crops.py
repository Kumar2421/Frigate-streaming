import os
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Query

from frigate.api.defs.tags import Tags
from frigate.const import CLIPS_DIR


router = APIRouter(tags=[Tags.media])


CropKind = Literal["first-person", "first-face"]
CropWhich = Literal["first", "best", "both"]


@dataclass(frozen=True)
class CropItem:
    kind: CropKind
    camera: str
    file: str
    path: str
    mtime: float


def _safe_relpath(p: Path, base: Path) -> str | None:
    try:
        rel = p.resolve().relative_to(base.resolve())
        return rel.as_posix()
    except Exception:
        return None


def _read_sidecar_metadata(image_path: Path) -> dict[str, Any] | None:
    try:
        sidecar = image_path.with_suffix(".json")
        if not sidecar.exists() or not sidecar.is_file():
            return None
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _list_crops(
    kind: CropKind,
    camera: str | None,
    which: CropWhich,
    before: float | None,
    limit: int,
) -> list[dict[str, Any]]:
    base = Path(CLIPS_DIR) / kind
    if not base.exists() or not base.is_dir():
        return []

    cameras: list[Path]
    if camera:
        cameras = [base / camera]
    else:
        cameras = [p for p in base.iterdir() if p.is_dir()]

    results: list[CropItem] = []

    for cam_dir in cameras:
        if not cam_dir.exists() or not cam_dir.is_dir():
            continue

        try:
            for p in cam_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name.lower()
                if not (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png") or name.endswith(".webp")):
                    continue

                if which != "both":
                    # Our naming convention includes '-first-' or '-best-'
                    # (with optional timestamp suffix)
                    if f"-{which}-" not in name:
                        continue

                try:
                    st = p.stat()
                except Exception:
                    continue

                results.append(
                    CropItem(
                        kind=kind,
                        camera=cam_dir.name,
                        file=p.name,
                        path=str(p),
                        mtime=float(st.st_mtime),
                    )
                )
        except Exception:
            continue

    # newest first
    results.sort(key=lambda x: x.mtime, reverse=True)

    if before is not None:
        results = [r for r in results if r.mtime < float(before)]

    results = results[:limit]

    # Build public URL via nginx /clips mount.
    # In Frigate, /clips maps to CLIPS_DIR.
    out: list[dict[str, Any]] = []
    for item in results:
        rel = _safe_relpath(Path(item.path), Path(CLIPS_DIR))
        if not rel:
            continue

        meta = _read_sidecar_metadata(Path(item.path))
        out.append(
            {
                "kind": item.kind,
                "camera": item.camera,
                "file": item.file,
                "mtime": item.mtime,
                "url": f"clips/{rel}",
                **(meta or {}),
            }
        )

    return out


@router.get("/crops/{kind}")
def list_crops(
    kind: CropKind,
    camera: str | None = None,
    which: CropWhich = Query(default="both"),
    before: float | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
):
    return {
        "kind": kind,
        "camera": camera,
        "which": which,
        "items": _list_crops(kind, camera, which, before, int(limit)),
    }
