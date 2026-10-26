"""Congela os artefatos atuais como uma iteração nomeada (ex.: v1).

Copia — nunca move — métricas, figuras, modelos e o relatório .md para
`outputs/iterations/<tag>/`, e grava um MANIFEST.json com o commit, o
`git status` e o SHA-256 de cada arquivo. Com isso dá para provar depois que a
comparação v1 × v2 usou exatamente os números que existiam antes da mudança.

Uso:
    python experiments/snapshot_iteration.py --tag v1
    python experiments/snapshot_iteration.py --tag v2 --seal   # iteração já gravada in-place
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITERATIONS_DIR = PROJECT_ROOT / "outputs" / "iterations"

# (origem, destino relativo dentro do snapshot)
SOURCES = [
    (PROJECT_ROOT / "outputs" / "metrics", "metrics"),
    (PROJECT_ROOT / "outputs" / "figures", "figures"),
    (PROJECT_ROOT / "outputs" / "models", "models"),
    (PROJECT_ROOT / "docs" / "resultados_tcc2.md", "docs/resultados_tcc2.md"),
]

IGNORED_PARTS = {"_smoke", "__pycache__"}
IGNORED_NAMES = {".gitkeep"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def iter_files(source: Path):
    if source.is_file():
        yield source
        return

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in IGNORED_NAMES:
            continue
        if IGNORED_PARTS.intersection(path.relative_to(source).parts):
            continue
        yield path


def write_manifest(target: Path, tag: str, files: dict[str, str], **extra) -> dict:
    manifest = {
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status": git("status", "--porcelain").splitlines(),
        **extra,
        "n_files": len(files),
        "files": files,
    }

    (target / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def seal(tag: str) -> None:
    """Congela in-place uma iteração que já foi gravada na própria pasta.

    A partir da v2 os experimentos escrevem direto em
    `outputs/iterations/<tag>/`, então não há o que copiar além do relatório
    .md: basta registrar o SHA-256 de cada arquivo que já está lá.
    """
    target = ITERATIONS_DIR / tag

    if not target.is_dir():
        raise SystemExit(f"Iteração '{tag}' não encontrada em {target}.")

    if (target / "MANIFEST.json").exists():
        raise SystemExit(
            f"A iteração '{tag}' já tem MANIFEST.json. "
            "Snapshots são imutáveis."
        )

    report = PROJECT_ROOT / "docs" / f"resultados_{tag}.md"

    if report.exists():
        out_path = target / "docs" / report.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, out_path)
    else:
        print(f"  [aviso] não encontrado, pulando: {report}")

    files = {
        path.relative_to(target).as_posix(): sha256(path)
        for path in iter_files(target)
    }

    manifest = write_manifest(target, tag, files, sealed_in_place=True)

    print(f"[ok] {len(files)} arquivos registrados em {target / 'MANIFEST.json'}")
    print(f"[ok] commit {manifest['git_commit'][:8]} "
          f"({len(manifest['git_status'])} arquivo(s) com mudanças não commitadas)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Nome da iteração, ex.: v1")
    parser.add_argument(
        "--seal",
        action="store_true",
        help="Congela in-place uma iteração já gravada em outputs/iterations/<tag>/ "
             "(copia só o docs/resultados_<tag>.md e grava o MANIFEST).",
    )
    args = parser.parse_args()

    if args.seal:
        seal(args.tag)
        return

    target = ITERATIONS_DIR / args.tag

    if target.exists():
        raise SystemExit(
            f"O snapshot '{args.tag}' já existe em {target}. "
            "Snapshots são imutáveis — escolha outra tag."
        )

    files: dict[str, str] = {}

    for source, destination in SOURCES:
        if not source.exists():
            print(f"  [aviso] não encontrado, pulando: {source}")
            continue

        for path in iter_files(source):
            if source.is_file():
                relative = Path(destination)
            else:
                relative = Path(destination) / path.relative_to(source)

            out_path = target / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)

            digest = sha256(path)
            if sha256(out_path) != digest:
                raise SystemExit(f"Cópia corrompida: {path}")

            files[relative.as_posix()] = digest

    manifest = write_manifest(target, args.tag, files)

    print(f"[ok] {len(files)} arquivos copiados para {target}")
    print(f"[ok] commit {manifest['git_commit'][:8]} "
          f"({len(manifest['git_status'])} arquivo(s) com mudanças não commitadas)")


if __name__ == "__main__":
    main()
