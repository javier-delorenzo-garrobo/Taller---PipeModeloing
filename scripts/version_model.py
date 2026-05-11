#!/usr/bin/env python3
"""Register and promote versioned model artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> Dict:
    if not path.exists():
        return {"active_version": None, "active_artifact": None, "versions": []}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(path: Path, registry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)
        file.write("\n")


def upsert_version(versions: List[Dict], entry: Dict, force: bool) -> List[Dict]:
    existing = next((item for item in versions if item.get("version") == entry["version"]), None)
    if existing and not force:
        raise SystemExit(
            f"La version {entry['version']} ya existe. Usa --force para reemplazarla."
        )

    filtered = [item for item in versions if item.get("version") != entry["version"]]
    filtered.append(entry)
    return sorted(filtered, key=lambda item: item["created_at"])


def run_git(args: List[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_git_repo() -> bool:
    try:
        return run_git(["rev-parse", "--is-inside-work-tree"]) == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def has_uncommitted_changes(paths: List[Path]) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *[str(path) for path in paths]],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def sync_with_git(version: str, paths: List[Path], tag_prefix: str) -> None:
    if not is_git_repo():
        raise SystemExit(
            "No se puede sincronizar con Git porque este directorio no es un repositorio. "
            "Ejecuta primero: git init"
        )

    tag_name = f"{tag_prefix}{version}"
    existing_tags = run_git(["tag", "--list", tag_name])
    if existing_tags:
        raise SystemExit(f"Ya existe el tag Git {tag_name}. Usa otra version.")

    run_git(["add", *[str(path) for path in paths]])
    if not has_uncommitted_changes(paths):
        print("No hay cambios nuevos que commitear en Git.")
        return

    run_git(["commit", "-m", f"Version model {version}"])
    run_git(["tag", "-a", tag_name, "-m", f"Model version {version}"])
    print(f"Commit y tag Git creados: {tag_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registra un modelo joblib en models/model_registry.json."
    )
    parser.add_argument(
        "--source",
        default="heart_disease_model.joblib",
        help="Ruta del modelo entrenado a registrar.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version semantica o etiqueta del modelo, por ejemplo 1.0.1.",
    )
    parser.add_argument(
        "--registry",
        default="models/model_registry.json",
        help="Ruta del registro JSON de modelos.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directorio donde guardar artefactos versionados.",
    )
    parser.add_argument(
        "--active-model",
        default="heart_disease_model.joblib",
        help="Ruta del modelo activo que carga la API.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Marca esta version como activa y actualiza el modelo que carga la API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza una version ya registrada.",
    )
    parser.add_argument(
        "--git",
        action="store_true",
        help="Hace git add, commit y tag del modelo versionado.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="model-v",
        help="Prefijo del tag Git que se crea con --git.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    registry_path = Path(args.registry)
    models_dir = Path(args.models_dir)
    active_model = Path(args.active_model)

    if not source.exists():
        raise SystemExit(f"No existe el modelo de origen: {source}")

    file_hash = sha256(source)
    short_hash = file_hash[:12]
    artifact_name = f"heart_disease_model_v{args.version}_{short_hash}.joblib"
    artifact_path = models_dir / artifact_name

    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, artifact_path)

    entry = {
        "version": args.version,
        "artifact": str(artifact_path),
        "sha256": file_hash,
        "sha256_short": short_hash,
        "size_bytes": source.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(source),
    }

    registry = load_registry(registry_path)
    registry["versions"] = upsert_version(registry.get("versions", []), entry, args.force)

    if args.promote:
        registry["active_version"] = args.version
        registry["active_artifact"] = str(artifact_path)
        if artifact_path.resolve() != active_model.resolve():
            shutil.copy2(artifact_path, active_model)

    save_registry(registry_path, registry)

    print(f"Registrado modelo v{args.version}: {artifact_path}")
    if args.promote:
        print(f"Version activa: {args.version} -> {active_model}")

    if args.git:
        git_paths = [artifact_path, registry_path]
        if args.promote:
            git_paths.append(active_model)
        sync_with_git(args.version, git_paths, args.tag_prefix)


if __name__ == "__main__":
    main()
