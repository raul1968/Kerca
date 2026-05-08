#!/usr/bin/env python3
"""Fix merged capsule embeddings by recomputing from their content."""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Kerca_kernel import bootstrap_kernel, deterministic_embed, ActivationState, ProcessingLane
import numpy as np

def main():
    kernel = bootstrap_kernel(load_existing=True)

    # Find all merged capsules (have + in name or have shadows)
    merged_caps = []
    for cid, cap in kernel.capsules.items():
        if cap.is_shadow or cap.merged_into:
            continue
        # Merged capsules: either have shadows, or name contains "+"
        if cap.shadows or "+" in cap.name:
            merged_caps.append(cid)

    print(f"Merged capsules found: {len(merged_caps)}")

    # Recompute embeddings from content keywords
    fixed = 0
    for cid in merged_caps:
        cap = kernel.capsules.get(cid)
        if not cap:
            continue

        # Build text from content to re-embed
        keywords = cap.content.get("keywords", [])
        description = cap.content.get("description", "")
        text = cap.name + " " + " ".join(keywords) + " " + str(description)[:500]
        
        new_emb = deterministic_embed(text)
        old_norm = float(np.linalg.norm(cap.embedding)) if cap.embedding is not None else 0
        
        cap.embedding = new_emb
        fixed += 1

    print(f"Fixed embeddings: {fixed}")

    # Now rebuild agreement graph with proper embeddings
    orbit2_ids = [cid for cid, c in kernel.capsules.items()
                  if c.orbit_level == 2 and not c.is_shadow and not c.merged_into]
    print(f"Orbit 2 capsules: {len(orbit2_ids)}")

    kernel.router.agreement_threshold = 0.25
    kernel.router.routing_rounds = 3

    batch_size = 400
    batches = [orbit2_ids[i:i+batch_size] for i in range(0, len(orbit2_ids), batch_size)]
    print(f"Processing {len(batches)} batches with fixed embeddings...")

    total_merges = 0
    for i, batch in enumerate(batches):
        batch_merges = 0
        for _ in range(2):
            kernel.capsules, merges, shadows = kernel.router.route(kernel.capsules, batch)
            batch_merges += len(merges)
            total_merges += len(merges)
        if batch_merges > 0:
            print(f"  Batch {i+1}/{len(batches)}: {batch_merges} merges")
        elif i % 5 == 0:
            print(f"  Batch {i+1}/{len(batches)}...")

    print(f"Total new merges: {total_merges}")

    # Boost gravity
    for cid in orbit2_ids:
        cap = kernel.capsules.get(cid)
        if cap and not cap.is_shadow and not cap.merged_into:
            cap.gravity_score = max(cap.gravity_score, 0.35)

    # Form clusters
    print("Forming clusters...")
    kernel.cluster_mgr.formation_threshold = 0.18
    clusters = kernel.cluster_mgr.form_clusters(
        kernel.capsules, kernel.router.agreement_graph, min_cluster_size=3
    )
    print(f"Clusters formed: {len(clusters)}")

    # Also try orbit 1+2+3 for cluster formation
    all_ids = [cid for cid, c in kernel.capsules.items()
               if c.orbit_level in (1, 2, 3) and not c.is_shadow and not c.merged_into]
    print(f"All eligible capsules: {len(all_ids)}")
    
    more_clusters = kernel.cluster_mgr.form_clusters(
        kernel.capsules, kernel.router.agreement_graph, min_cluster_size=5
    )
    print(f"Additional clusters (min size 5): {len(more_clusters)}")

    # Name clusters
    for cluster in kernel.cluster_mgr.clusters.values():
        if cluster.centroid_capsule_id:
            centroid = kernel.capsules.get(cluster.centroid_capsule_id)
            if centroid:
                cluster.name = centroid.name[:50]

    total_clusters = len(kernel.cluster_mgr.clusters)
    print(f"Total clusters: {total_clusters}")

    if total_clusters > 0:
        # Activate all clusters
        print("Activating clusters...")
        for cid in list(kernel.cluster_mgr.clusters.keys()):
            kernel.cluster_mgr.activate_cluster(cid, sustain=4)

        for _ in range(5):
            r = kernel.run_cycle()
            print(f"  Cycle {r['cycle']}: {r['active_clusters']} clusters, "
                  f"{r['merges_this_cycle']} merges, "
                  f"{r['tensions_created']} tensions, "
                  f"{r['ergo_resolutions']} ergo")
    else:
        # Desperate measure: force clusters from keyword overlap directly
        print("\nNo clusters from agreement graph. Trying direct keyword clustering...")
        force_keyword_clusters(kernel)

    # Save
    kernel.router.agreement_threshold = 0.63
    kernel.store.save(kernel.capsules, clusters=kernel.cluster_mgr.to_dict())

    final = kernel.status()
    print(f"\n{'=' * 60}")
    print(f"FINAL STATE")
    print(f"{'=' * 60}")
    print(f"  Orbit 0: {final['orbit_distribution'].get(0, 0)}")
    print(f"  Orbit 1: {final['orbit_distribution'].get(1, 0)}")
    print(f"  Orbit 2: {final['orbit_distribution'].get(2, 0)}")
    print(f"  Orbit 3: {final['orbit_distribution'].get(3, 0)}")
    print(f"  Clusters: {final['total_clusters']}")
    print(f"  Active clusters: {final['active_clusters']}")
    print(f"  Tensions: {final['tensions']}")
    print(f"  Total merges: {final['routing_stats']['merges_performed']}")
    print(f"\n  Saved. Launch GUI: python kerca_oracle_gui.py")


def force_keyword_clusters(kernel):
    """Build clusters directly from keyword overlap when agreement graph fails."""
    from collections import defaultdict
    from Kerca_kernel import Cluster
    
    # Get all orbit 2 capsules
    caps = [(cid, c) for cid, c in kernel.capsules.items()
            if c.orbit_level == 2 and not c.is_shadow and not c.merged_into]
    
    # Extract dominant keywords per capsule
    cap_keywords = {}
    for cid, cap in caps:
        keywords = set(cap.content.get("keywords", []))
        cap_keywords[cid] = keywords
    
    # Group by shared keywords
    keyword_groups: Dict[str, set] = defaultdict(set)
    for cid, keywords in cap_keywords.items():
        for kw in keywords:
            if len(kw) > 4:  # Only substantive keywords
                keyword_groups[kw].add(cid)
    
    # Form clusters from keyword groups with enough overlap
    cluster_id = 0
    used_capsules = set()
    
    for kw, cids in sorted(keyword_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(cids) < 3:
            continue
        unused = cids - used_capsules
        if len(unused) < 3:
            continue
        
        cluster_id += 1
        cluster_name = f"KW: {kw}"
        
        # Find centroid
        best_cid = None
        best_grav = -1
        for cid in unused:
            cap = kernel.capsules.get(cid)
            if cap and cap.gravity_score > best_grav:
                best_grav = cap.gravity_score
                best_cid = cid
        
        cluster = Cluster(
            id=f"kw_cluster_{cluster_id}",
            name=cluster_name,
            capsule_ids=unused,
            centroid_capsule_id=best_cid,
            activation_level=0.0,
        )
        kernel.cluster_mgr.clusters[cluster.id] = cluster
        
        for cid in unused:
            if cid in kernel.capsules:
                kernel.capsules[cid].cluster_id = cluster.id
        
        used_capsules.update(unused)
    
    print(f"  Keyword clusters formed: {cluster_id}")


if __name__ == "__main__":
    main()