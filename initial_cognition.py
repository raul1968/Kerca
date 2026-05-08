#!/usr/bin/env python3
"""
KERCA Initial Cognition Push
=============================
Bootstraps the KERCA knowledge graph after PDF ingestion.

Process:
  1. Warm all capsules to READY state
  2. Lower routing threshold temporarily to build agreement graph
  3. Run refinement cycles at normal threshold
  4. Force cluster formation
  5. Cascade activation to build inter-cluster edges
  6. Trigger ergoregion reconciliation
  7. Final cleanup and save

Designed for large-scale ingestion (200+ papers).
"""

import sys
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kerca_kernel import (
    bootstrap_kernel,
    ActivationState,
    ProcessingLane,
)

LOW_THRESHOLD = 0.30
NORMAL_THRESHOLD = 0.63
REFINEMENT_CYCLES = 5
CASCADE_CYCLES = 3
FINAL_CYCLES = 5
MIN_CLUSTER_SIZE = 3
BATCH_SIZE = 500


def main():
    print("=" * 70)
    print("KERCA Initial Cognition Push")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n[0] Loading kernel...")
    kernel = bootstrap_kernel(load_existing=True)

    before_active = sum(1 for c in kernel.capsules.values()
                       if not c.is_shadow and not c.merged_into)
    before_shadows = sum(1 for c in kernel.capsules.values() if c.is_shadow)

    print(f"  Capsules: {before_active} active, {before_shadows} shadows")
    print(f"  Cycles already run: {kernel.cycle_count}")

    for orbit in range(4):
        count = sum(1 for c in kernel.capsules.values()
                   if c.orbit_level == orbit and not c.is_shadow and not c.merged_into)
        print(f"  Orbit {orbit}: {count}")

    # Phase 1: Warm
    print("\n[Phase 1] Warming capsules...")
    warmed = 0
    for cap in kernel.capsules.values():
        if cap.is_shadow or cap.merged_into:
            continue
        if cap.orbit_level == 3:
            cap.activation_state = ActivationState.READY
            cap.gravity_score = max(cap.gravity_score, 0.20)
            cap.processing_lane = ProcessingLane.INGESTION
            warmed += 1
        elif cap.orbit_level == 2:
            cap.activation_state = ActivationState.READY
            cap.gravity_score = max(cap.gravity_score, 0.30)
            warmed += 1
        elif cap.orbit_level == 1:
            cap.activation_state = ActivationState.READY
            warmed += 1
    print(f"  {warmed} capsules warmed")

    # Phase 2: Low threshold routing
    print(f"\n[Phase 2] Building agreement graph (threshold lowered to {LOW_THRESHOLD})...")
    kernel.router.agreement_threshold = LOW_THRESHOLD

    all_ids = [cid for cid, c in kernel.capsules.items()
               if not c.is_shadow and not c.merged_into]
    batches = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    print(f"  Processing {len(all_ids)} capsules in {len(batches)} batches")

    phase2_merges = 0
    phase2_shadows = 0

    for batch_num, batch in enumerate(batches):
        print(f"  Batch {batch_num+1}/{len(batches)} ({len(batch)} capsules)...", end=" ")
        batch_merges = 0
        batch_shadows = 0
        for _ in range(2):
            kernel.capsules, merges, shadows = kernel.router.route(kernel.capsules, batch)
            batch_merges += len(merges)
            batch_shadows += len(shadows)
        phase2_merges += batch_merges
        phase2_shadows += batch_shadows
        kernel.gravity_orbit.update_gravity(kernel.capsules, batch, kernel.cycle_count)
        kernel.cycle_count += 1
        print(f"{batch_merges} merges, {batch_shadows} shadows")

    print(f"  Phase 2 total: {phase2_merges} merges, {phase2_shadows} shadows")

    # Phase 3: Refinement
    print(f"\n[Phase 3] Refinement cycles (threshold restored to {NORMAL_THRESHOLD})...")
    kernel.router.agreement_threshold = NORMAL_THRESHOLD
    phase3_merges = 0
    phase3_tensions = 0

    for _ in range(REFINEMENT_CYCLES):
        result = kernel.run_cycle()
        phase3_merges += result['merges_this_cycle']
        phase3_tensions += result['tensions_created']
        print(f"  Cycle {result['cycle']}: "
              f"{result['merges_this_cycle']} merges, "
              f"{result['shadows_created']} shadows, "
              f"{result['tensions_created']} tensions, "
              f"{result['active_clusters']} clusters, "
              f"max gravity: {result['max_gravity']:.3f}")

    print(f"  Phase 3 total: {phase3_merges} merges, {phase3_tensions} tensions")

    # Phase 4: Cluster formation
    print(f"\n[Phase 4] Forming clusters (min size: {MIN_CLUSTER_SIZE})...")
    kernel.cluster_mgr.formation_threshold = 0.25
    new_clusters = kernel.cluster_mgr.form_clusters(
        kernel.capsules, kernel.router.agreement_graph, min_cluster_size=MIN_CLUSTER_SIZE
    )
    print(f"  Clusters formed: {len(new_clusters)}")

    named = 0
    for cluster in kernel.cluster_mgr.clusters.values():
        if cluster.centroid_capsule_id:
            centroid = kernel.capsules.get(cluster.centroid_capsule_id)
            if centroid:
                cluster.name = centroid.name[:50]
                named += 1
    print(f"  Clusters named: {named}")

    # Phase 5: Cascade
    print(f"\n[Phase 5] Cascade activation...")
    cluster_ids = list(kernel.cluster_mgr.clusters.keys())
    for i, cid in enumerate(cluster_ids):
        if i % 20 == 0:
            print(f"  Activating cluster {i+1}/{len(cluster_ids)}...")
        kernel.cluster_mgr.activate_cluster(cid, sustain=2)

    phase5_total_active = 0
    for i in range(CASCADE_CYCLES):
        result = kernel.run_cycle()
        phase5_total_active += result['active_clusters']
        print(f"  Cascade cycle {i+1}: {result['active_clusters']} clusters active, "
              f"{result['merges_this_cycle']} merges")
    print(f"  Phase 5: avg {phase5_total_active / max(1, CASCADE_CYCLES):.1f} clusters per cycle")

    # Phase 6: Final cycles with KERCA protections
    print(f"\n[Phase 6] Final cycles with ergoregion reconciliation...")
    phase6_merges = 0
    phase6_tensions = 0
    phase6_ergo = 0
    phase6_reproc = 0
    phase6_challenges = 0

    for _ in range(FINAL_CYCLES):
        result = kernel.run_cycle()
        phase6_merges += result['merges_this_cycle']
        phase6_tensions += result['tensions_created']
        phase6_ergo += result['ergo_resolutions']
        phase6_reproc += result['reprocessed']
        phase6_challenges += result['challenges_created']
        print(f"  Cycle {result['cycle']}: "
              f"{result['merges_this_cycle']} merges, "
              f"{result['tensions_created']} tensions, "
              f"{result['ergo_resolutions']} ergo, "
              f"{result['reprocessed']} reproc, "
              f"{result['challenges_created']} challenges")

    print(f"  Phase 6 totals: {phase6_merges} merges, {phase6_tensions} tensions, "
          f"{phase6_ergo} ergo, {phase6_reproc} reproc, {phase6_challenges} challenges")

    # Phase 7: Cool down
    print(f"\n[Phase 7] Cooling to resting state...")
    kernel.cluster_mgr.decay_all()
    cooled = 0
    for cap in kernel.capsules.values():
        if cap.activation_state == ActivationState.ACTIVE:
            cap.activation_state = ActivationState.COOLING
            cap.sustain_cycles = 0
            cooled += 1
        elif cap.activation_state == ActivationState.COOLING and cap.sustain_cycles <= 0:
            cap.activation_state = ActivationState.READY
            cooled += 1
    print(f"  {cooled} capsules cooled")

    # Save
    print(f"\n[Phase 8] Saving...")
    kernel.store.save(
        kernel.capsules, clusters=kernel.cluster_mgr.to_dict(),
        extra={"cycle_count": kernel.cycle_count,
               "routing_stats": kernel.router.routing_stats,
               "timeline_stats": kernel.timeline.stats(),
               "cognition_push_completed": datetime.now().isoformat()}
    )
    print(f"  Saved to: {kernel.store.store_path}")

    # Final report
    final = kernel.status()
    after_active = sum(1 for c in kernel.capsules.values()
                      if not c.is_shadow and not c.merged_into)
    after_shadows = sum(1 for c in kernel.capsules.values() if c.is_shadow)

    print(f"\n{'=' * 70}")
    print(f"COGNITION PUSH COMPLETE")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    print(f"\n  Capsules: {before_active} → {after_active} active")
    print(f"  Shadows: {before_shadows} → {after_shadows}")
    print(f"  Orbit dist: {final['orbit_distribution']}")
    print(f"  Clusters: {final['total_clusters']}")
    print(f"  Merges: {final['routing_stats']['merges_performed']}")
    print(f"  Tensions: {final['tensions']}")
    print(f"\nNext: python promote_merged.py")


if __name__ == "__main__":
    main()