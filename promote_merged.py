#!/usr/bin/env python3
"""
KERCA Orbit Promotion Pass
===========================
After initial cognition, merged capsules at orbit 3 need help climbing.
This script:
  1. Identifies merged capsules at orbit 3 with gravity > 0.25
  2. Promotes them to orbit 2 (Working memory)
  3. Boosts their gravity slightly
  4. Re-runs routing with orbit-appropriate thresholds
  5. Forces cluster formation on the promoted capsules
"""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Kerca_kernel import bootstrap_kernel, ActivationState, ProcessingLane

def main():
    print("=" * 60)
    print("KERCA Orbit Promotion Pass")
    print("=" * 60)
    
    kernel = bootstrap_kernel(load_existing=True)
    
    # Stats before
    orbit3_before = sum(1 for c in kernel.capsules.values() 
                       if c.orbit_level == 3 and not c.is_shadow and not c.merged_into)
    orbit2_before = sum(1 for c in kernel.capsules.values() 
                       if c.orbit_level == 2 and not c.is_shadow and not c.merged_into)
    print(f"\nBefore: Orbit 3={orbit3_before}, Orbit 2={orbit2_before}")
    
    # Find merged capsules at orbit 3 with decent gravity
    promoted = 0
    for cid, cap in list(kernel.capsules.items()):
        if cap.is_shadow or cap.merged_into:
            continue
        if cap.orbit_level != 3:
            continue
        
        # Promote if: it's a merged capsule OR has gravity > 0.25
        is_merged = "+" in cap.name and "merged_" not in cid  # has been merged (name contains +)
        has_gravity = cap.gravity_score > 0.25
        
        if is_merged or has_gravity:
            cap.orbit_level = 2
            cap.orbit_radius = 0.55
            cap.gravity_score = max(cap.gravity_score, 0.35)
            cap.activation_state = ActivationState.READY
            cap.processing_lane = ProcessingLane.COGNITIVE
            promoted += 1
    
    print(f"Promoted {promoted} capsules to Orbit 2")
    
    # Also promote high-gravity non-merged capsules
    high_grav = 0
    for cid, cap in kernel.capsules.items():
        if cap.is_shadow or cap.merged_into:
            continue
        if cap.orbit_level == 3 and cap.gravity_score >= 0.35:
            cap.orbit_level = 2
            cap.orbit_radius = 0.55
            cap.activation_state = ActivationState.READY
            cap.processing_lane = ProcessingLane.COGNITIVE
            high_grav += 1
    
    print(f"Promoted {high_grav} additional high-gravity capsules to Orbit 2")
    
    # Now route at Orbit 2 threshold (0.55 for orbit 1-1 pairs, 0.63 default)
    print("\nRunning routing on promoted capsules...")
    orbit2_ids = [cid for cid, c in kernel.capsules.items() 
                  if c.orbit_level == 2 and not c.is_shadow and not c.merged_into]
    print(f"  {len(orbit2_ids)} capsules at Orbit 2")
    
    # Lower threshold temporarily for Orbit 2 consolidation
    kernel.router.agreement_threshold = 0.50
    kernel.router.routing_rounds = 3
    
    total_merges = 0
    for _ in range(3):  # 3 cycles
        kernel.capsules, merges, shadows = kernel.router.route(kernel.capsules, orbit2_ids)
        total_merges += len(merges)
        for cid in orbit2_ids:
            cap = kernel.capsules.get(cid)
            if cap and not cap.is_shadow and not cap.merged_into:
                kernel.gravity_orbit.boost_gravity(cap, amount=0.08)
        kernel.gravity_orbit.update_gravity(kernel.capsules, orbit2_ids)
        kernel.cycle_count += 1
        print(f"  Cycle: {len(merges)} merges, {len(shadows)} shadows")
    
    print(f"  Total merges: {total_merges}")
    kernel.router.agreement_threshold = 0.63
    
    # Force cluster formation on orbit 1+2 capsules
    print("\nForming clusters from Orbit 1-2 capsules...")
    eligible = [cid for cid, c in kernel.capsules.items()
                if c.orbit_level in (1, 2) and not c.is_shadow and not c.merged_into]
    print(f"  {len(eligible)} eligible capsules")
    
    kernel.cluster_mgr.formation_threshold = 0.20
    clusters = kernel.cluster_mgr.form_clusters(
        kernel.capsules,
        kernel.router.agreement_graph,
        min_cluster_size=3
    )
    print(f"  {len(clusters)} clusters formed")
    
    # Name clusters
    for cluster in kernel.cluster_mgr.clusters.values():
        if cluster.centroid_capsule_id:
            centroid = kernel.capsules.get(cluster.centroid_capsule_id)
            if centroid:
                cluster.name = centroid.name[:50]
    
    # Cascade activate
    print("\nCascade activation...")
    for cid in list(kernel.cluster_mgr.clusters.keys()):
        kernel.cluster_mgr.activate_cluster(cid, sustain=4)
    
    for _ in range(3):
        result = kernel.run_cycle()
        print(f"  Cycle {result['cycle']}: {result['active_clusters']} clusters, "
              f"{result['merges_this_cycle']} merges, {result['tensions_created']} tensions")
    
    # Save
    kernel.store.save(kernel.capsules, clusters=kernel.cluster_mgr.to_dict())
    
    # Final stats
    final = kernel.status()
    print(f"\n{'=' * 60}")
    print(f"PROMOTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Orbit 0: {final['orbit_distribution'].get(0, 0)}")
    print(f"  Orbit 1: {final['orbit_distribution'].get(1, 0)}")
    print(f"  Orbit 2: {final['orbit_distribution'].get(2, 0)}")
    print(f"  Orbit 3: {final['orbit_distribution'].get(3, 0)}")
    print(f"  Clusters: {final['total_clusters']}")
    print(f"  Tensions: {final['tensions']}")
    print(f"  Challenges: {final['challenges']}")
    print(f"\nLaunch GUI: python kerca_oracle_gui.py")

if __name__ == "__main__":
    main()