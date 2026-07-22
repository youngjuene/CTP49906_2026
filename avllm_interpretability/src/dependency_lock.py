"""Build and verify a fail-closed static dependency/runtime identity candidate."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import re
import sysconfig
from typing import Any, Iterable, Mapping


TARGET_DEPENDENCY_LOCK_SCHEMA = "target-dependency-lock/2.0.0"
REQUIRED_IDENTITY_BINDINGS = frozenset(
    {
        "repo_baseline",
        "model",
        "processor",
        "tokenizer",
        "chat_template",
        "notebook_dependencies",
        "prompts",
        "stimulus_manifest",
        "replay",
        "precompute",
    }
)
REQUIRED_FFMPEG_LIBRARIES = frozenset(
    {
        "libavcodec",
        "libavdevice",
        "libavfilter",
        "libavformat",
        "libavutil",
        "libswresample",
        "libswscale",
    }
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME_RE = re.compile(r"[-_.]+")
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")
_PEP723_DEPENDENCY_RE = re.compile(r'^#\s+"([^"]+)"')


@dataclass(frozen=True)
class LockVerification:
    identity_valid: bool
    source_requirements_valid: bool | None
    candidate_valid: bool
    target_ready: bool
    issues: tuple[str, ...]
    target_blockers: tuple[str, ...]
    computed_lock_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_valid": self.identity_valid,
            "source_requirements_valid": self.source_requirements_valid,
            "candidate_valid": self.candidate_valid,
            "target_ready": self.target_ready,
            "issues": list(self.issues),
            "target_blockers": list(self.target_blockers),
            "computed_lock_identity": self.computed_lock_identity,
        }


def normalize_package_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("package name must be non-empty")
    return _NAME_RE.sub("-", name.strip()).lower()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_object_identity(value: Any) -> str:
    return "sha256:" + sha256(_canonical_bytes(value)).hexdigest()


def canonical_lock_bytes(lock: Mapping[str, Any]) -> bytes:
    if not isinstance(lock, Mapping):
        raise TypeError("lock must be a mapping")
    return _canonical_bytes(
        {str(key): value for key, value in lock.items() if key != "lock_identity"}
    )


def compute_lock_identity(lock: Mapping[str, Any]) -> str:
    return "sha256:" + sha256(canonical_lock_bytes(lock)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_manifest(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifest = []
    for path in sorted({item.resolve() for item in paths}, key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = path.name
        manifest.append(
            {"path": relative, "sha256": _file_sha256(path), "size": path.stat().st_size}
        )
    return manifest


def _binding(kind: str, content: Any, sources: Iterable[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "identity": canonical_object_identity(content),
        "sources": sorted(set(sources)),
        "content": content,
    }


def parse_direct_requirements(requirements_bytes: bytes) -> set[str]:
    direct = set()
    for raw_line in requirements_bytes.decode("utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        match = _REQUIREMENT_RE.match(line)
        if match is None:
            raise ValueError(f"requirement is not exactly pinned: {raw_line}")
        direct.add(normalize_package_name(match.group(1)))
    if not direct:
        raise ValueError("requirements contain no direct packages")
    return direct


def parse_pep723_dependencies(notebook_source: str) -> tuple[str, tuple[str, ...], set[str]]:
    lines = notebook_source.splitlines()
    if not lines or lines[0].strip() != "# /// script":
        raise ValueError("notebook is missing a leading PEP 723 block")
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "# ///"
        )
    except StopIteration as exc:
        raise ValueError("notebook PEP 723 block is not closed") from exc
    block = "\n".join(lines[: closing_index + 1]) + "\n"
    specs = tuple(
        match.group(1)
        for line in block.splitlines()
        if (match := _PEP723_DEPENDENCY_RE.match(line)) is not None
    )
    if not specs:
        raise ValueError("PEP 723 block contains no dependencies")
    names = set()
    for spec in specs:
        match = re.match(r"^([A-Za-z0-9_.-]+)", spec)
        if match is None:
            raise ValueError(f"invalid PEP 723 dependency: {spec}")
        names.add(normalize_package_name(match.group(1)))
    return block, specs, names


def required_distribution_closure(
    root_names: Iterable[str], *, distributions: Iterable[Any] | None = None
) -> set[str]:
    """Resolve the locally installed course closure with environment markers."""

    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
    except ImportError as exc:
        raise RuntimeError("packaging is required to derive the installed closure") from exc
    source = list(metadata.distributions() if distributions is None else distributions)
    by_name = {
        normalize_package_name(distribution.metadata["Name"]): distribution
        for distribution in source
    }
    pending = [normalize_package_name(name) for name in root_names]
    closure: set[str] = set()
    environment = default_environment()
    environment["extra"] = ""
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        distribution = by_name.get(name)
        if distribution is None:
            raise ValueError(f"required distribution missing from environment: {name}")
        closure.add(name)
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = normalize_package_name(requirement.name)
            if dependency not in closure:
                pending.append(dependency)
    return closure


def _record_manifest(record_text: str) -> list[dict[str, Any]]:
    entries = []
    for row in csv.reader(record_text.splitlines()):
        if not row:
            continue
        entries.append(
            {
                "path": row[0],
                "record_hash": row[1] if len(row) > 1 and row[1] else None,
                "size": int(row[2]) if len(row) > 2 and row[2] else None,
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def installed_distribution_inventory(
    *, direct_names: Iterable[str] = (), distributions: Iterable[Any] | None = None
) -> list[dict[str, Any]]:
    """Fingerprint every installed distribution from its deterministic RECORD."""

    direct = {normalize_package_name(name) for name in direct_names}
    source = metadata.distributions() if distributions is None else distributions
    inventory: list[dict[str, Any]] = []
    for distribution in source:
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise ValueError("installed distribution is missing Name metadata")
        name = normalize_package_name(raw_name)
        record = distribution.read_text("RECORD")
        if not record:
            raise ValueError(f"installed distribution {name} has no RECORD")
        record_manifest = _record_manifest(record)
        if not record_manifest:
            raise ValueError(f"installed distribution {name} has an empty RECORD")
        inventory.append(
            {
                "name": name,
                "version": str(distribution.version),
                "direct": name in direct,
                "record_entry_count": len(record_manifest),
                "installed_record_sha256": "sha256:"
                + sha256(record.encode("utf-8")).hexdigest(),
                "installed_tree_sha256": canonical_object_identity(record_manifest),
                "installed_tree_fingerprint_kind": "normalized-installed-RECORD",
                "wheel_artifact_sha256": None,
            }
        )
    inventory.sort(key=lambda item: str(item["name"]))
    names = [str(item["name"]) for item in inventory]
    if len(names) != len(set(names)):
        raise ValueError("authorized environment contains duplicate normalized distributions")
    missing_direct = sorted(direct - set(names))
    if missing_direct:
        raise ValueError(f"direct requirements missing from environment: {missing_direct}")
    return inventory


def local_runtime_identity() -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to fingerprint linked FFmpeg") from exc
    linked = {
        name: list(version) for name, version in sorted(av.library_versions.items())
    }
    missing = REQUIRED_FFMPEG_LIBRARIES - set(linked)
    if missing:
        raise RuntimeError(f"PyAV linked FFmpeg identity is incomplete: {sorted(missing)}")
    libc_name, libc_version = platform.libc_ver()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "platform_tag": sysconfig.get_platform(),
        "machine": platform.machine(),
        "libc": {"name": libc_name, "version": libc_version},
        "pyav_version": str(av.__version__),
        "linked_ffmpeg": linked,
    }


def _hf_blob_manifest(snapshot: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    manifest = []
    for name in sorted(set(names)):
        path = snapshot / name
        if not path.exists():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        manifest.append(
            {
                "path": name,
                "snapshot_object": resolved.name,
                "size": path.stat().st_size,
            }
        )
    return manifest


def build_static_candidate_lock(
    *, repo_root: Path, snapshot_root: Path, baseline_commit: str
) -> dict[str, Any]:
    """Build the strongest local candidate without making a target claim."""

    repo_root = repo_root.resolve()
    avllm = repo_root / "avllm_interpretability"
    requirements = (avllm / "requirements.txt").read_bytes()
    direct = parse_direct_requirements(requirements)
    notebook_source = (avllm / "CTP49906_avllm_molab.py").read_text(encoding="utf-8")
    pep723_block, pep723_specs, pep723_names = parse_pep723_dependencies(notebook_source)
    environment_distributions = list(metadata.distributions())
    distributions = installed_distribution_inventory(
        direct_names=direct, distributions=environment_distributions
    )
    course_roots = direct | pep723_names | {"av"}
    closure_names = required_distribution_closure(
        course_roots, distributions=environment_distributions
    )
    closure_inventory = [
        item for item in distributions if item["name"] in closure_names
    ]
    runtime = local_runtime_identity()
    meta = json.loads((avllm / "precomputed/meta.json").read_text(encoding="utf-8"))

    model_files = ["config.json", "generation_config.json", "model.safetensors.index.json"]
    model_files.extend(
        path.name for path in snapshot_root.glob("model-*.safetensors")
    )
    processor_files = ["preprocessor_config.json", "config.json", "spk_dict.pt"]
    tokenizer_files = [
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    prompt_content = {
        "logit_prompt": meta["logit_prompt"],
        "attention_prompt": meta["attention_prompt"],
        "knockout_rules": meta["knockout_rules"],
        "max_new_tokens": meta["max_new_tokens"],
    }
    assets = _file_manifest(
        repo_root,
        [avllm / "assets/02321.mp4", avllm / "assets/02321_silent.mp4"],
    )
    stimulus_content = {
        "status": "LOCAL_CANDIDATE_NOT_RELEASED",
        "selected_clip": meta["clip"],
        "assets": assets,
    }
    replay_paths = [
        repo_root / "study_materials/wp6/replay_manifest.json",
        avllm / "src/precompute.py",
        avllm / "precomputed/README.md",
    ]
    precompute_paths = [
        path for path in (avllm / "precomputed").iterdir() if path.is_file()
    ]
    bindings = {
        "repo_baseline": _binding(
            "git-commit", {"commit": baseline_commit}, ["git baseline"]
        ),
        "model": _binding(
            "hf-snapshot-blob-references",
            {
                "model_id": meta["model"],
                "revision": meta["model_revision"],
                "files": _hf_blob_manifest(snapshot_root, model_files),
            },
            [f"hf://{meta['model']}@{meta['model_revision']}"],
        ),
        "processor": _binding(
            "hf-snapshot-files",
            _file_manifest(snapshot_root, [snapshot_root / name for name in processor_files]),
            processor_files,
        ),
        "tokenizer": _binding(
            "hf-snapshot-files",
            _file_manifest(snapshot_root, [snapshot_root / name for name in tokenizer_files]),
            tokenizer_files,
        ),
        "chat_template": _binding(
            "hf-snapshot-file",
            _file_manifest(snapshot_root, [snapshot_root / "chat_template.json"]),
            ["chat_template.json"],
        ),
        "notebook_dependencies": _binding(
            "pep723-script-metadata",
            {
                "block_sha256": "sha256:"
                + sha256(pep723_block.encode("utf-8")).hexdigest(),
                "requires_python": ">=3.10",
                "dependencies": list(pep723_specs),
                "pyav_runtime_package": "av",
            },
            ["avllm_interpretability/CTP49906_avllm_molab.py#PEP-723"],
        ),
        "prompts": _binding(
            "canonical-prompt-contract", prompt_content, ["precomputed/meta.json"]
        ),
        "stimulus_manifest": _binding(
            "local-candidate-stimulus-manifest",
            stimulus_content,
            [item["path"] for item in assets],
        ),
        "replay": _binding(
            "replay-contract-files",
            _file_manifest(repo_root, replay_paths),
            [str(path.relative_to(repo_root)) for path in replay_paths],
        ),
        "precompute": _binding(
            "precompute-artifact-set",
            _file_manifest(repo_root, precompute_paths),
            [str(path.relative_to(repo_root)) for path in precompute_paths],
        ),
    }
    lock: dict[str, Any] = {
        "schema_version": TARGET_DEPENDENCY_LOCK_SCHEMA,
        "status": "STATIC_CANDIDATE_UNVERIFIED",
        "readiness": False,
        "scope": "local-authorized-avllm-environment",
        "inventory_role": "local_environment_observation_not_target_dependency_closure",
        "source_requirements": {
            "path": "avllm_interpretability/requirements.txt",
            "sha256": "sha256:" + sha256(requirements).hexdigest(),
            "direct_packages": sorted(direct),
        },
        "distribution_count": len(distributions),
        "distribution_inventory_identity": canonical_object_identity(distributions),
        "installed_distributions": distributions,
        "course_required_closure": {
            "root_packages": sorted(course_roots),
            "distribution_count": len(closure_inventory),
            "distribution_names": sorted(closure_names),
            "inventory_identity": canonical_object_identity(closure_inventory),
        },
        "local_runtime": runtime,
        "local_runtime_identity": canonical_object_identity(runtime),
        "target_runtime": {
            "python_abi": None,
            "platform_tag": None,
            "runtime_image_digest": None,
            "pyav_version": None,
            "linked_ffmpeg": None,
        },
        "bindings": bindings,
        "limitations": [
            "Installed RECORD/tree fingerprints are not wheel artifact hashes.",
            "Exact target-Molab runtime and wheel artifacts were not observed.",
            "The stimulus binding is a local candidate manifest, not a licensed release manifest.",
        ],
    }
    lock["lock_identity"] = compute_lock_identity(lock)
    return lock


def _package_map(items: Any, issues: list[str]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(items, list) or not items:
        issues.append("installed_distributions must be a non-empty list")
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(f"installed distribution at index {index} must be an object")
            continue
        raw_name = item.get("name")
        if not isinstance(raw_name, str):
            issues.append(f"installed distribution at index {index} has invalid name")
            continue
        name = normalize_package_name(raw_name)
        if name in result:
            issues.append(f"duplicate normalized package name: {name}")
        result[name] = item
    return result


def verify_target_dependency_lock(
    lock: Mapping[str, Any],
    *,
    requirements_bytes: bytes | None = None,
    observed_installed_distributions: Mapping[str, Mapping[str, Any]] | None = None,
    observed_runtime: Mapping[str, Any] | None = None,
    observed_bindings: Mapping[str, str] | None = None,
) -> LockVerification:
    if not isinstance(lock, Mapping):
        raise TypeError("lock must be a mapping")
    issues: list[str] = []
    blockers: list[str] = []
    computed = compute_lock_identity(lock)
    identity_valid = lock.get("lock_identity") == computed
    if not identity_valid:
        issues.append("lock_identity does not match canonical lock content")
    if lock.get("schema_version") != TARGET_DEPENDENCY_LOCK_SCHEMA:
        issues.append("unsupported target dependency lock schema")
    if lock.get("readiness") is not False:
        issues.append("static candidate readiness must be false")
    if lock.get("inventory_role") != (
        "local_environment_observation_not_target_dependency_closure"
    ):
        issues.append("installed inventory role is missing or misleading")

    source = lock.get("source_requirements")
    source_valid: bool | None = None
    direct_expected: set[str] = set()
    if not isinstance(source, Mapping):
        issues.append("source_requirements must be an object")
    else:
        if not _is_sha256(source.get("sha256")):
            issues.append("source_requirements.sha256 is missing or invalid")
        if requirements_bytes is not None:
            observed_hash = "sha256:" + sha256(requirements_bytes).hexdigest()
            source_valid = source.get("sha256") == observed_hash
            if not source_valid:
                issues.append("source requirements hash does not match lock")
            try:
                direct_expected = parse_direct_requirements(requirements_bytes)
            except ValueError as exc:
                issues.append(str(exc))

    packages = _package_map(lock.get("installed_distributions"), issues)
    if lock.get("distribution_count") != len(packages):
        issues.append("distribution_count does not match installed_distributions")
    if lock.get("distribution_inventory_identity") != canonical_object_identity(
        lock.get("installed_distributions")
    ):
        issues.append("distribution_inventory_identity does not match inventory")
    for name, item in packages.items():
        if not isinstance(item.get("version"), str) or not item["version"]:
            issues.append(f"distribution {name} is missing version")
        for field in ("installed_record_sha256", "installed_tree_sha256"):
            if not _is_sha256(item.get(field)):
                issues.append(f"distribution {name} has invalid {field}")
        if not isinstance(item.get("record_entry_count"), int) or item["record_entry_count"] < 1:
            issues.append(f"distribution {name} has invalid record_entry_count")
    if direct_expected:
        flagged = {name for name, item in packages.items() if item.get("direct") is True}
        for name in sorted(direct_expected - set(packages)):
            issues.append(f"direct requirement missing from lock: {name}")
        if flagged != direct_expected:
            issues.append("direct package flags disagree with requirements")

    closure = lock.get("course_required_closure")
    closure_names: set[str] = set()
    if not isinstance(closure, Mapping):
        issues.append("course_required_closure must be an object")
    else:
        raw_closure_names = closure.get("distribution_names")
        if not isinstance(raw_closure_names, list) or not all(
            isinstance(name, str) for name in raw_closure_names
        ):
            issues.append("course_required_closure.distribution_names is invalid")
        else:
            closure_names = {normalize_package_name(name) for name in raw_closure_names}
            missing_closure = sorted(closure_names - set(packages))
            if missing_closure:
                issues.append(
                    f"course closure distributions missing from inventory: {missing_closure}"
                )
            closure_inventory = [
                item
                for item in lock.get("installed_distributions", [])
                if isinstance(item, Mapping) and item.get("name") in closure_names
            ]
            if closure.get("distribution_count") != len(closure_inventory):
                issues.append("course closure distribution_count mismatch")
            if closure.get("inventory_identity") != canonical_object_identity(
                closure_inventory
            ):
                issues.append("course closure inventory_identity mismatch")
        raw_roots = closure.get("root_packages")
        if not isinstance(raw_roots, list) or not all(
            isinstance(name, str) for name in raw_roots
        ):
            issues.append("course_required_closure.root_packages is invalid")
        elif direct_expected and not direct_expected <= {
            normalize_package_name(name) for name in raw_roots
        }:
            issues.append("course closure roots omit direct requirements")
    for name in sorted(closure_names & set(packages)):
        if not _is_sha256(packages[name].get("wheel_artifact_sha256")):
            blockers.append(f"distribution {name} wheel_artifact_sha256 is UNVERIFIED")

    if observed_installed_distributions is not None:
        observed = {
            normalize_package_name(name): value
            for name, value in observed_installed_distributions.items()
        }
        for name in sorted(set(observed) - set(packages)):
            issues.append(f"installed distribution missing from lock: {name}")
        for name in sorted(set(packages) - set(observed)):
            issues.append(f"locked distribution missing from observed environment: {name}")
        for name in sorted(set(packages) & set(observed)):
            for field in ("version", "installed_record_sha256", "installed_tree_sha256"):
                if packages[name].get(field) != observed[name].get(field):
                    issues.append(f"installed distribution mismatch: {name}.{field}")

    runtime = lock.get("local_runtime")
    if not isinstance(runtime, Mapping):
        issues.append("local_runtime must be an object")
    else:
        if lock.get("local_runtime_identity") != canonical_object_identity(runtime):
            issues.append("local_runtime_identity does not match local_runtime metadata")
        linked = runtime.get("linked_ffmpeg")
        if not isinstance(linked, Mapping) or set(linked) != REQUIRED_FFMPEG_LIBRARIES:
            issues.append("local runtime PyAV linked FFmpeg identity is incomplete")
        if not isinstance(runtime.get("pyav_version"), str) or not runtime["pyav_version"]:
            issues.append("local runtime PyAV version is missing")
        if observed_runtime is not None and canonical_object_identity(runtime) != canonical_object_identity(observed_runtime):
            issues.append("observed runtime does not match locked runtime metadata")

    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping):
        issues.append("bindings must be an object")
        bindings = {}
    for name in sorted(REQUIRED_IDENTITY_BINDINGS):
        binding_item = bindings.get(name)
        if not isinstance(binding_item, Mapping):
            issues.append(f"missing identity binding: {name}")
            continue
        if not _is_sha256(binding_item.get("identity")):
            issues.append(f"identity binding is unhashed: {name}")
        elif binding_item["identity"] != canonical_object_identity(
            binding_item.get("content")
        ):
            issues.append(f"identity binding content disagreement: {name}")
        if observed_bindings is not None and observed_bindings.get(
            name
        ) != binding_item.get("identity"):
            issues.append(f"identity mismatch: {name}")

    target_runtime = lock.get("target_runtime")
    if not isinstance(target_runtime, Mapping):
        blockers.append("target_runtime is UNVERIFIED")
    else:
        for field in ("python_abi", "platform_tag", "runtime_image_digest", "pyav_version", "linked_ffmpeg"):
            if not target_runtime.get(field):
                blockers.append(f"target_runtime.{field} is UNVERIFIED")

    candidate_valid = identity_valid and source_valid is not False and not issues
    target_ready = (
        candidate_valid
        and lock.get("status") == "LOCKED"
        and not blockers
        and lock.get("readiness") is True
    )
    return LockVerification(
        identity_valid=identity_valid,
        source_requirements_valid=source_valid,
        candidate_valid=candidate_valid,
        target_ready=target_ready,
        issues=tuple(issues),
        target_blockers=tuple(blockers),
        computed_lock_identity=computed,
    )


__all__ = [
    "LockVerification",
    "REQUIRED_IDENTITY_BINDINGS",
    "TARGET_DEPENDENCY_LOCK_SCHEMA",
    "build_static_candidate_lock",
    "canonical_lock_bytes",
    "canonical_object_identity",
    "compute_lock_identity",
    "installed_distribution_inventory",
    "local_runtime_identity",
    "normalize_package_name",
    "parse_direct_requirements",
    "verify_target_dependency_lock",
]
