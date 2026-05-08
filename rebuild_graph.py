#!/usr/bin/env python3
"""Rebuild agreement graph for Orbit 2 capsules and force cluster formation."""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Kerca_kernel import bootstrap_kernel

def main():
    kernel = bootstrap_kernel(load_existing=True)

    orbit2_ids = [cid for cid, c in kernel.capsules.items()
                  if c.orbit_level == 2 and not c.is_shadow and not c.merged_into]
    print(f"Orbit 2 capsules: {len(orbit2_ids)}")

    kernel.router.agreement_threshold = 0.25
    kernel.router.routing_rounds = 3

    batch_size = 400
    batches = [orbit2_ids[i:i+batch_size] for i in range(0, len(orbit2_ids), batch_size)]
    print(f"Processing {len(batches)} batches...")

    total_merges = 0
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)} ({len(batch)} caps)...", end=" ")
        batch_merges = 0
        for _ in range(2):
            kernel.capsules, merges, shadows = kernel.router.route(kernel.capsules, batch)
            batch_merges += len(merges)
            total_merges += len(merges)
        print(f"{batch_merges} merges")

    print(f"Total new merges: {total_merges}")

    print("Forming clusters...")
    kernel.cluster_mgr.formation_threshold = 0.20
    clusters = kernel.cluster_mgr.form_clusters(
        kernel.capsules, kernel.router.agreement_graph, min_cluster_size=3
    )
    print(f"Clusters formed: {len(clusters)}")

    for cluster in kernel.cluster_mgr.clusters.values():
        if cluster.centroid_capsule_id:
            centroid = kernel.capsules.get(cluster.centroid_capsule_id)
            if centroid:
                cluster.name = centroid.name[:50]

    for cid in list(kernel.cluster_mgr.clusters.keys()):
        kernel.cluster_mgr.activate_cluster(cid, sustain=4)

    for _ in range(3):
        r = kernel.run_cycle()
        print(f"  Cycle: {r['active_clusters']} clusters, "
              f"{r['merges_this_cycle']} merges, "
              f"{r['tensions_created']} tensions")

    kernel.router.agreement_threshold = 0.63
    kernel.store.save(kernel.capsules, clusters=kernel.cluster_mgr.to_dict())

    final = kernel.status()
    print(f"\nFinal:")
    print(f"  Clusters: {final['total_clusters']}")
    print(f"  Active clusters: {final['active_clusters']}")
    print(f"  Tensions: {final['tensions']}")
    print(f"  Orbit dist: {final['orbit_distribution']}")
    print(f"  Total merges: {final['routing_stats']['merges_performed']}")
    print(f"\nSaved. Launch GUI: python kerca_oracle_gui.py")

if __name__ == "__main__":
    main()