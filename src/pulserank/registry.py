import hashlib
import json
import os
import shutil
from pathlib import Path

from .ml.model import ModelBundle


class PromotionError(ValueError):
    pass


class ModelRegistry:
    """Filesystem model registry with quality gates and atomic promotion."""

    def __init__(self, root="artifacts/registry"):
        self.root = Path(root)
        self.models = self.root / "models"
        self.state_path = self.root / "registry.json"

    def state(self):
        if not self.state_path.exists():
            return {"format_version": 1, "active_version": None, "previous_version": None, "versions": {}}
        return json.loads(self.state_path.read_text())

    def register(self, artifact):
        artifact = Path(artifact)
        bundle = ModelBundle.load(artifact)
        report = json.loads((artifact / "evaluation.json").read_text())
        if report["model_version"] != bundle.version:
            raise PromotionError("manifest and evaluation model versions do not match")
        destination = self.models / bundle.version
        if destination.exists():
            if self._checksums(destination) != self._checksums(artifact):
                raise PromotionError(f"model version {bundle.version} already exists with different files")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.models / f".{bundle.version}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(artifact, temporary)
            os.replace(temporary, destination)
        gates = self.evaluate_gates(report)
        state = self.state()
        state["versions"][bundle.version] = {
            "status": "champion" if state["active_version"] == bundle.version else "staged",
            "path": str(destination.resolve()), "checksums": self._checksums(destination),
            "gates": gates,
        }
        self._write(state)
        return {"model_version": bundle.version, "gates": gates, "path": str(destination)}

    @staticmethod
    def evaluate_gates(report, minimum_ndcg_lift=0.02):
        learned = report["metrics"]["learned_ranker"]
        baseline = report["metrics"]["heuristic"]
        checks = {
            "ndcg_lift": learned["ndcg_at_5"] - baseline["ndcg_at_5"] >= minimum_ndcg_lift,
            "recall_non_regression": learned["recall_at_5"] >= baseline["recall_at_5"],
            "coverage_non_regression": learned["coverage_at_5"] >= baseline["coverage_at_5"],
            "training_converged": report["training"]["final_bpr_loss"] < report["training"]["initial_bpr_loss"],
        }
        return {"passed": all(checks.values()), "checks": checks,
                "minimum_ndcg_lift": minimum_ndcg_lift}

    def promote(self, version):
        state = self.state()
        entry = state["versions"].get(version)
        if entry is None:
            raise PromotionError(f"unknown model version: {version}")
        if not entry["gates"]["passed"]:
            failed = [name for name, passed in entry["gates"]["checks"].items() if not passed]
            raise PromotionError(f"promotion gates failed: {', '.join(failed)}")
        if self._checksums(Path(entry["path"])) != entry["checksums"]:
            raise PromotionError("artifact checksum validation failed")
        old = state["active_version"]
        if old and old in state["versions"]:
            state["versions"][old]["status"] = "previous"
        entry["status"] = "champion"
        state["previous_version"] = old
        state["active_version"] = version
        self._write(state)
        return self.status()

    def rollback(self):
        state = self.state()
        previous = state["previous_version"]
        if previous is None:
            raise PromotionError("no previous model is available for rollback")
        current = state["active_version"]
        state["versions"][current]["status"] = "staged"
        state["versions"][previous]["status"] = "champion"
        state["active_version"], state["previous_version"] = previous, current
        self._write(state)
        return self.status()

    def active_path(self):
        state = self.state()
        version = state["active_version"]
        return Path(state["versions"][version]["path"]) if version else None

    def status(self):
        state = self.state()
        return {
            "active_version": state["active_version"], "previous_version": state["previous_version"],
            "registered_versions": len(state["versions"]),
            "active_path": str(self.active_path()) if self.active_path() else None,
        }

    def _write(self, state):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)

    @staticmethod
    def _checksums(directory):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir()) if path.is_file()
        }
