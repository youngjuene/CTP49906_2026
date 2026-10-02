"""Restore the Studio assets carried by a single uploaded notebook.

The function is self-contained so the build script can copy it into the notebook
without requiring this module to exist in a hosted checkout first.
"""


def restore_studio_bundle(notebook_dir, payload, expected_digest):
    """Install missing/unchanged owned files; refuse to overwrite edited files."""
    import base64
    import hashlib
    import json
    import zlib
    from pathlib import Path, PurePosixPath

    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(payload, validate=True), 512 * 1024)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Studio bundle is truncated or exceeds 512 KiB")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("Studio bundle digest mismatch")
    document = json.loads(raw)
    if document.get("schema") != 1 or not isinstance(document.get("files"), dict):
        raise ValueError("Unsupported Studio bundle schema")
    files = document["files"]
    for name, content in files.items():
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute() or ".." in parts
                or "\\" in name or not isinstance(content, str)):
            raise ValueError(f"Unsafe Studio bundle path: {name!r}")

    base = Path(notebook_dir).resolve()
    root = base / "studio" / "ctp49906-kr"
    manifest = root / ".bundle-manifest.json"
    # Refuse symlinks even inside the workspace; uploaded views are plain files.
    for destination in [manifest, *(root / name for name in files)]:
        for candidate in [destination, *destination.parents]:
            if candidate == base:
                break
            if candidate.is_symlink():
                raise ValueError(f"Studio bundle destination is a symlink: {candidate}")
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    pending, conflicts = [], []
    for name in files:
        path = root / name
        if not path.exists():
            pending.append(name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == hashes[name]:
            continue
        if actual == previous.get("files", {}).get(name):
            pending.append(name)
        else:
            conflicts.append(name)
    if conflicts:
        raise FileExistsError("Studio files have local edits; preserve or back them up before updating: "
                              + ", ".join(conflicts))
    for name in pending:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[name], encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema": 1, "bundle": expected_digest, "files": hashes}
    if previous != state:
        manifest.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"root": str(root), "written": pending, "bundle": expected_digest}
