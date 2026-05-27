"""Quick diagnostic for collision_index.json — checks the index isn't empty
or under-populated. CPU only, seconds.

    PYTHONPATH=src python scripts/eval/inspect_collision_index.py \
        data/processed/collision_index.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: inspect_collision_index.py <path-to-collision_index.json>")
    path = Path(sys.argv[1])
    with path.open("r", encoding="utf-8") as handle:
        idx = json.load(handle)

    print(f"file: {path}")
    print(f"entries (CAM_FRONT keyframes): {len(idx)}")

    future_lens: list[int] = []
    agent_counts_per_step: list[int] = []
    empty_steps = 0
    cat_counts: dict[str, int] = {}
    samples_with_any_agent = 0
    samples_all_empty = 0

    for entry in idx.values():
        futures = entry.get("future_samples", [])
        future_lens.append(len(futures))
        any_agent = False
        for step in futures:
            agents = step.get("agents", [])
            agent_counts_per_step.append(len(agents))
            if agents:
                any_agent = True
            else:
                empty_steps += 1
            for a in agents:
                cat = a.get("cat", "?")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if any_agent:
            samples_with_any_agent += 1
        else:
            samples_all_empty += 1

    if future_lens:
        print(f"\nfuture-step counts per entry:")
        print(f"  avg {sum(future_lens)/len(future_lens):.2f},  min {min(future_lens)},  max {max(future_lens)}")
        zero_futures = sum(1 for n in future_lens if n == 0)
        print(f"  entries with 0 futures: {zero_futures}")

    if agent_counts_per_step:
        total_agent_records = sum(agent_counts_per_step)
        n_steps = len(agent_counts_per_step)
        print(f"\nagent counts per future step ({n_steps} step records total):")
        print(f"  avg {total_agent_records/n_steps:.2f},  min {min(agent_counts_per_step)},  max {max(agent_counts_per_step)}")
        print(f"  empty steps: {empty_steps} ({empty_steps/n_steps:.1%})")
        print(f"  total agent records kept: {total_agent_records}")

    print(f"\nentries with at least one agent in any future step: {samples_with_any_agent}")
    print(f"entries with ZERO agents across all futures: {samples_all_empty}")

    print(f"\ntop categories kept:")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {cat:40s} {n}")

    print(f"\nfirst entry preview:")
    first_key = next(iter(idx))
    preview = idx[first_key]
    futures = preview.get("future_samples", [])
    print(f"  fname: {first_key}")
    print(f"  sample_token: {preview.get('sample_token','?')}")
    print(f"  n_futures: {len(futures)}")
    if futures:
        first_future = futures[0]
        agents = first_future.get("agents", [])
        print(f"  future[0] dt={first_future.get('dt')} agents={len(agents)}")
        for a in agents[:5]:
            print(f"    {a}")


if __name__ == "__main__":
    main()
