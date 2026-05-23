import sys
import copy
import yaml
from pathlib import Path


def set_dotted(d, key, val):
    *parents, last = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[last] = val


def main():
    sweep_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = yaml.safe_load(sweep_path.read_text())

    main_dir = Path(__file__).resolve().parent.parent
    base_rel = sweep.get("base", "args/parameters.yaml")
    base_path = (main_dir / base_rel) if not Path(base_rel).is_absolute() else Path(base_rel)
    base = yaml.safe_load(base_path.read_text())

    for run in sweep["runs"]:
        cfg = copy.deepcopy(base)
        for k, v in run.get("overrides", {}).items():
            set_dotted(cfg, k, v)
        cfg["run_name"] = run["name"]
        out = out_dir / f"{sweep_path.stem}__{run['name']}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(out)


if __name__ == "__main__":
    main()
