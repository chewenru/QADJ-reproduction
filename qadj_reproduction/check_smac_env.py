import json
from importlib import metadata
from pathlib import Path


EXPECTED_SC2_BUILD = "Base69232"
EXPECTED_SC2_VERSION = "4.6.2.69232"
EXPECTED_SMAC_COMMIT = "d6aab33f76abc3849c50463a8592a84f59a5ef84"


def _read_direct_url(dist_name: str):
    dist = metadata.distribution(dist_name)
    direct_url = None
    for file in dist.files or []:
        if str(file).endswith("direct_url.json"):
            direct_url = Path(dist.locate_file(file))
            break
    if direct_url and direct_url.exists():
        return json.loads(direct_url.read_text())
    return None


def _sc2_info(repo_root: Path):
    sc2_root = repo_root / "3rdparty" / "StarCraftII"
    versions_dir = sc2_root / "Versions"
    builds = sorted(p.name for p in versions_dir.glob("Base*")) if versions_dir.exists() else []
    return {
        "sc2_root": str(sc2_root),
        "builds": builds,
        "expected_build": EXPECTED_SC2_BUILD,
        "expected_version": EXPECTED_SC2_VERSION,
        "matches_expected": EXPECTED_SC2_BUILD in builds,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    smac_dist = metadata.distribution("smac")
    pysc2_dist = metadata.distribution("pysc2")
    smac_direct_url = _read_direct_url("smac") or {}
    smac_commit = smac_direct_url.get("vcs_info", {}).get("commit_id")

    report = {
        "smac": {
            "version": smac_dist.version,
            "commit": smac_commit,
            "matches_expected_commit": smac_commit == EXPECTED_SMAC_COMMIT,
        },
        "pysc2": {
            "version": pysc2_dist.version,
        },
        "sc2": _sc2_info(repo_root),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
