#!/usr/bin/env python3
"""Activate all clusters and run KERCA cognition cycles."""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Kerca_kernel import bootstrap_kernel, ActivationState

kernel = bootstrap_kernel(load_existing=True)

print(f"Clusters: {len(kernel.cluster_mgr.clusters)}")
print(f"Orbit 2 caps: {sum(1 for c in kernel.capsules.values() if c.orbit_level==2 and not c.is_shadow)}")

# Show cluster names
for cid, cluster in list(kernel.cluster_mgr.clusters.items())[:10]:
    print(f"  {cluster.name}: {len(cluster.capsule_ids)} capsules")

# Activate all
cluster_ids = list(kernel.cluster_mgr.clusters.keys())
print(f"\nActivating {len(cluster_ids)} clusters...")
for cid in cluster_ids:
    kernel.cluster_mgr.activate_cluster(cid, sustain=5)

# Run cycles
print()
for i in range(8):
    r = kernel.run_cycle()
    print(f"Cycle {r['cycle']}: "
          f"{r['active_clusters']} clusters, "
          f"{r['merges_this_cycle']} merges, "
          f"{r['tensions_created']} tensions, "
          f"{r['ergo_resolutions']} ergo, "
          f"{r['reprocessed']} reproc")

# Cool
kernel.cluster_mgr.decay_all()
for c in kernel.capsules.values():
    if c.activation_state == ActivationState.ACTIVE:
        c.activation_state = ActivationState.COOLING
        c.sustain_cycles = 0

kernel.store.save(kernel.capsules, clusters=kernel.cluster_mgr.to_dict())

final = kernel.status()
print(f"\nFinal:")
print(f"  Clusters: {final['total_clusters']} ({final['active_clusters']} active)")
print(f"  Tensions: {final['tensions']}")
print(f"  Challenges: {final['challenges']}")
print(f"  Orbits: {final['orbit_distribution']}")
print(f"  Merges total: {final['routing_stats']['merges_performed']}")
print(f"\nLaunch GUI: python kerca_oracle_gui.py")