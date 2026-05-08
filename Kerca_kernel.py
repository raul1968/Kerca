#!/usr/bin/env python3
"""
KERCA — Kerr Engine for Routing and Capsule Agreement  v1.0
=============================================================
Deterministic capsule-based cognitive engine with rotational feedback
and comprehensive risk mitigations.

Architecture:
  • Semantic Capsules with gravity, orbit, and activation state
  • Routing-by-Agreement (deterministic, no gradients)
  • Cluster-based chain-lightning activation
  • Ergoregion reconciliation (forced near-miss resolution)
  • Temporal reprocessing (bounded re-evaluation of past states)
  • Tension capsules (active contradiction recording)
  • Core inertia constraint (Orbit 0 multi-cycle consensus)

Risk Mitigations (baked into core, not bolted on):
  1. Feedback explosion → Reprocessing budget (max 3) + decay factor
  2. Tension proliferation → Significance gate + tension merging + stale resolution
  3. Orbit 0 contamination → Core inertia + external challenges + multi-cycle consensus

Author: KERCA Development
Date:   2026-05-07
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Sequence, Set, Tuple, Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# 0.  Project root & paths
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent

PROJECT_ROOT: Path = _resolve_project_root()
JSON_DIR: Path = PROJECT_ROOT / "Json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KNOWLEDGE_BASE_PATH: Path = JSON_DIR / "kerca_knowledge_base.json"
DEFAULT_TEMPORAL_STORE_PATH: Path = JSON_DIR / "kerca_timeline.jsonl"

# ---------------------------------------------------------------------------
# 1.  Core Enums
# ---------------------------------------------------------------------------

class CapsuleKind(str, Enum):
    NUCLEUS      = "nucleus"       # Core identity capsule
    TOPIC        = "topic"         # Knowledge topic
    SKILL        = "skill"         # Capability
    MEMORY       = "memory"        # Episodic/log
    WORKFLOW     = "workflow"      # Process template
    TENSION      = "tension"       # Unresolved contradiction (KERCA)
    CHALLENGE    = "challenge"     # Orbit 0 challenge record (KERCA)
    EXPERIMENTAL = "experimental"

class ActivationState(str, Enum):
    COLD       = "cold"            # Inactive, may hibernate
    READY      = "ready"           # Warm, available for activation
    ACTIVE     = "active"          # Currently firing
    COOLING    = "cooling"         # Recently active, winding down

class ProcessingLane(str, Enum):
    INGESTION  = "ingestion"       # Raw data intake
    COGNITIVE  = "cognitive"       # Active reasoning / merge
    ARCHIVE    = "archive"         # Long-term memory
    ERGO       = "ergo"            # Ergoregion reconciliation (KERCA)

ORBIT_LABELS: Dict[int, str] = {
    0: "Core",
    1: "Active",
    2: "Working",
    3: "Ingest",
}

LANE_BOUNDARIES: Dict[int, Tuple[float, float]] = {
    0: (0.0,  0.20),
    1: (0.20, 0.45),
    2: (0.45, 0.72),
    3: (0.72, 1.00),
}

# ---------------------------------------------------------------------------
# 2.  Capsule (KERCA v1.0 — with risk mitigation fields baked in)
# ---------------------------------------------------------------------------

@dataclass
class Capsule:
    """Single KERCA capsule.

    Risk mitigation fields (baked into core):
      reprocess_count / max_reprocess → feedback explosion cap
      tension_pairs → tracked contradictions
      core_inertia / core_consensus_required → orbit 0 protection
    """
    id: str
    kind: CapsuleKind
    name: str

    # --- Core identity ---
    state: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    usage_count: int = 0

    # --- Temporal ---
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    last_activated_frame: int = 0

    # --- Orbital / gravity ---
    orbit_level: int = 2
    orbit_radius: float = 0.60
    orbit_angle: float = 0.0
    gravity_score: float = 0.1
    agreement_score: float = 0.5

    # --- Activation ---
    activation_state: ActivationState = ActivationState.COLD
    processing_lane: ProcessingLane = ProcessingLane.ARCHIVE
    sustain_cycles: int = 0
    ready_triggered_by: List[str] = field(default_factory=list)

    # --- Merge / shadow ---
    is_shadow: bool = False
    lineage: List[str] = field(default_factory=list)
    shadows: List[str] = field(default_factory=list)
    merged_into: Optional[str] = None
    merge_confidence: float = 0.0

    # --- Cluster ---
    cluster_id: Optional[str] = None

    # --- KERCA: Reprocessing budget (Risk 1 mitigation) ---
    reprocess_count: int = 0
    max_reprocess: int = 3
    reprocess_decay: float = 0.7
    reprocess_exhausted: bool = False

    # --- KERCA: Tension tracking (Risk 2 mitigation) ---
    tension_pairs: List[str] = field(default_factory=list)
    needs_review: bool = False

    # --- KERCA: Core inertia (Risk 3 mitigation) ---
    core_inertia: int = 0
    core_consensus_required: int = 5
    core_agreements: List[str] = field(default_factory=list)
    core_modification_count: int = 0

    # --- Ergo processing ---
    ergo_processed: bool = False

    # --- Embedding ---
    embedding: Optional[np.ndarray] = None

    # --- Content ---
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["activation_state"] = self.activation_state.value
        d["processing_lane"] = self.processing_lane.value
        d["created_at"] = self.created_at.isoformat()
        d["last_used_at"] = self.last_used_at.isoformat()
        d["embedding"] = self.embedding.tolist() if self.embedding is not None else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capsule":
        data = dict(data)
        data["kind"] = CapsuleKind(data["kind"]) if data.get("kind") else CapsuleKind.TOPIC
        if "activation_state" in data and isinstance(data["activation_state"], str):
            data["activation_state"] = ActivationState(data["activation_state"])
        else:
            data["activation_state"] = ActivationState.COLD
        if "processing_lane" in data and isinstance(data["processing_lane"], str):
            data["processing_lane"] = ProcessingLane(data["processing_lane"])
        else:
            data["processing_lane"] = ProcessingLane.ARCHIVE
        for key in ("created_at", "last_used_at"):
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        if data.get("embedding") is not None:
            data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------
    #  Agreement & merge
    # ------------------------------------------------------------------

    def evaluate_agreement(self, other: "Capsule") -> float:
        """Pairwise agreement score. Orbit 0 capsules exert weighted influence."""
        if self.embedding is not None and other.embedding is not None:
            norm_a = float(np.linalg.norm(self.embedding))
            norm_b = float(np.linalg.norm(other.embedding))
            if norm_a > 1e-9 and norm_b > 1e-9:
                base_score = float(np.dot(self.embedding, other.embedding) / (norm_a * norm_b))
            else:
                base_score = 0.1
        else:
            kw_a = set(self.content.get("keywords", []))
            kw_b = set(other.content.get("keywords", []))
            if kw_a or kw_b:
                union = kw_a | kw_b
                base_score = len(kw_a & kw_b) / len(union) if union else 0.0
            else:
                base_score = 0.3 if self.kind == other.kind else 0.1

        if self.orbit_level == 0 or other.orbit_level == 0:
            base_score = min(1.0, base_score * 1.2)
        return base_score

    def get_merge_threshold(self, other: "Capsule") -> float:
        """Per-pair merge threshold with orbit-aware graduated scale."""
        if self.orbit_level == 0 and other.orbit_level == 0:
            return 0.85
        elif self.orbit_level == 0 or other.orbit_level == 0:
            return 0.75
        elif self.orbit_level == 1 and other.orbit_level == 1:
            return 0.55
        elif self.orbit_level == 3 and other.orbit_level == 3:
            return 0.40  # Ingestion capsules merge more freely
        return 0.63

    def merge(self, other: "Capsule") -> "Capsule":
        """Deterministic merge. Originals become shadows."""
        merged_id = f"merged_{hashlib.sha256(f'{self.id}+{other.id}'.encode()).hexdigest()[:12]}"
        total_usage = self.usage_count + other.usage_count
        merged_conf = (
            (self.usage_count / total_usage) * self.confidence + 
            (other.usage_count / total_usage) * other.confidence
        ) if total_usage > 0 else (self.confidence + other.confidence) / 2.0

        emb = None
        if self.embedding is not None and other.embedding is not None:
            emb = (self.embedding * self.usage_count + other.embedding * other.usage_count) / max(total_usage, 1)
        else:
            emb = self.embedding or other.embedding

        merged_state = dict(self.state)
        for k, v in other.state.items():
            if k not in merged_state:
                merged_state[k] = v
            elif merged_state[k] != v:
                merged_state[k] = [merged_state[k], v]

        merged_content = dict(self.content)
        merged_content.setdefault("keywords", [])
        merged_content["keywords"] = list(
            set(merged_content["keywords"]) | set(other.content.get("keywords", []))
        )
        merged_content["original_contents"] = [self.content, other.content]

        orbit_level = self.orbit_level if self.gravity_score >= other.gravity_score else other.orbit_level

        if ProcessingLane.ERGO in (self.processing_lane, other.processing_lane):
            proc_lane = ProcessingLane.ERGO
        elif ProcessingLane.COGNITIVE in (self.processing_lane, other.processing_lane):
            proc_lane = ProcessingLane.COGNITIVE
        elif ProcessingLane.INGESTION in (self.processing_lane, other.processing_lane):
            proc_lane = ProcessingLane.INGESTION
        else:
            proc_lane = ProcessingLane.ARCHIVE

        # KERCA: inherit reprocess budget state
        merged_reprocess_count = max(self.reprocess_count, other.reprocess_count)
        merged_reprocess_exhausted = self.reprocess_exhausted or other.reprocess_exhausted

        # KERCA: inherit core inertia
        merged_core_inertia = max(self.core_inertia, other.core_inertia)
        merged_core_mod_count = self.core_modification_count + other.core_modification_count

        return Capsule(
            id=merged_id, kind=self.kind,
            name=f"{self.name}+{other.name}",
            state=merged_state, confidence=merged_conf,
            usage_count=self.usage_count + other.usage_count,
            orbit_level=orbit_level,
            gravity_score=max(self.gravity_score, other.gravity_score),
            agreement_score=(self.agreement_score + other.agreement_score) / 2.0,
            activation_state=ActivationState.ACTIVE,
            processing_lane=proc_lane,
            sustain_cycles=4,
            lineage=sorted(set(self.lineage + other.lineage + [self.id, other.id])),
            shadows=[self.id, other.id],
            merge_confidence=merged_conf,
            embedding=emb,
            content=merged_content,
            metadata={"merged_from": [self.id, other.id], "merged_at": datetime.now().isoformat()},
            created_at=min(self.created_at, other.created_at),
            last_used_at=max(self.last_used_at, other.last_used_at),
            ergo_processed=self.ergo_processed or other.ergo_processed,
            tension_pairs=list(set(self.tension_pairs + other.tension_pairs)),
            reprocess_count=merged_reprocess_count,
            max_reprocess=max(self.max_reprocess, other.max_reprocess),
            reprocess_decay=min(self.reprocess_decay, other.reprocess_decay),
            reprocess_exhausted=merged_reprocess_exhausted,
            core_inertia=merged_core_inertia,
            core_consensus_required=max(self.core_consensus_required, other.core_consensus_required),
            core_modification_count=merged_core_mod_count,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def touch(self, frame_number: int = 0) -> None:
        self.last_used_at = datetime.now()
        self.last_activated_frame = frame_number
        self.usage_count += 1

    def to_dict_safe(self) -> Dict[str, Any]:
        return self.to_dict()

    def is_active(self) -> bool:
        return (
            not self.is_shadow 
            and not self.merged_into 
            and self.activation_state == ActivationState.ACTIVE
        )

    def is_available(self) -> bool:
        return (
            not self.is_shadow
            and not self.merged_into
            and self.activation_state in (
                ActivationState.READY, ActivationState.ACTIVE, ActivationState.COOLING
            )
        )

    def can_reprocess(self) -> bool:
        """KERCA Risk 1: check if capsule still has reprocessing budget."""
        return (
            not self.reprocess_exhausted 
            and self.reprocess_count < self.max_reprocess
            and self.orbit_level != 0  # Orbit 0 never reprocessed directly
        )


# ---------------------------------------------------------------------------
# 3.  Cluster System
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """Semantic cluster of capsules that activate together."""
    id: str
    name: str
    capsule_ids: Set[str] = field(default_factory=set)
    centroid_capsule_id: Optional[str] = None
    activation_level: float = 0.0
    sustain_cycles: int = 0
    cooldown_rate: float = 0.25
    trigger_threshold: float = 0.55
    cascade_threshold: float = 0.4
    max_sustain_cycles: int = 8
    created_at: datetime = field(default_factory=datetime.now)
    last_fired_at: Optional[datetime] = None
    fire_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fire(self, sustain_cycles: int = 5) -> None:
        self.activation_level = 1.0
        self.sustain_cycles = min(sustain_cycles, self.max_sustain_cycles)
        self.last_fired_at = datetime.now()
        self.fire_count += 1

    def decay(self) -> None:
        if self.sustain_cycles > 0:
            self.sustain_cycles -= 1
        if self.sustain_cycles <= 0:
            self.activation_level = max(0.0, self.activation_level - self.cooldown_rate)

    def cascade_to(self, other: "Cluster", edge_strength: float) -> float:
        if self.activation_level < self.cascade_threshold:
            return 0.0
        return self.activation_level * edge_strength * 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "capsule_ids": list(self.capsule_ids),
            "centroid_capsule_id": self.centroid_capsule_id,
            "activation_level": self.activation_level,
            "sustain_cycles": self.sustain_cycles,
            "cooldown_rate": self.cooldown_rate,
            "trigger_threshold": self.trigger_threshold,
            "cascade_threshold": self.cascade_threshold,
            "max_sustain_cycles": self.max_sustain_cycles,
            "created_at": self.created_at.isoformat(),
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
            "fire_count": self.fire_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        data = dict(data)
        data["capsule_ids"] = set(data.get("capsule_ids", []))
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        else:
            data["created_at"] = datetime.now()
        if data.get("last_fired_at"):
            data["last_fired_at"] = datetime.fromisoformat(data["last_fired_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ClusterManager:
    """Cluster formation, activation, cascade triggering."""

    def __init__(self) -> None:
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_edges: Dict[Tuple[str, str], float] = {}
        self.formation_threshold: float = 0.5

    def form_clusters(
        self, capsules: Dict[str, Capsule],
        agreement_graph: Dict[str, Dict[str, float]],
        min_cluster_size: int = 3,
    ) -> List[Cluster]:
        active_capsules = {
            cid: cap for cid, cap in capsules.items()
            if not cap.is_shadow and not cap.merged_into
        }
        edges: List[Tuple[str, str, float]] = []
        for cid_a, neighbors in agreement_graph.items():
            for cid_b, score in neighbors.items():
                if cid_a < cid_b and score >= self.formation_threshold:
                    if cid_a in active_capsules and cid_b in active_capsules:
                        edges.append((cid_a, cid_b, score))
        edges.sort(key=lambda x: x[2], reverse=True)

        parent: Dict[str, str] = {}
        def find(x: str) -> str:
            if x not in parent: parent[x] = x
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]
        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry

        for cid_a, cid_b, _score in edges:
            union(cid_a, cid_b)

        groups: Dict[str, Set[str]] = defaultdict(set)
        for cid in active_capsules:
            groups[find(cid)].add(cid)

        new_clusters: List[Cluster] = []
        for root, capsule_set in groups.items():
            if len(capsule_set) >= min_cluster_size:
                existing = self._find_existing_cluster(capsule_set)
                if existing:
                    existing.capsule_ids.update(capsule_set)
                    new_clusters.append(existing)
                else:
                    cluster = Cluster(
                        id=f"cluster_{hashlib.md5(str(sorted(capsule_set)).encode()).hexdigest()[:10]}",
                        name=f"Cluster-{len(self.clusters)+1}",
                        capsule_ids=capsule_set,
                        centroid_capsule_id=self._find_centroid(capsule_set, capsules),
                    )
                    self.clusters[cluster.id] = cluster
                    new_clusters.append(cluster)

        self._update_cluster_edges(agreement_graph, capsules)
        for cluster in new_clusters:
            for cid in cluster.capsule_ids:
                if cid in capsules:
                    capsules[cid].cluster_id = cluster.id
        return new_clusters

    def _find_existing_cluster(self, capsule_set: Set[str]) -> Optional[Cluster]:
        for cluster in self.clusters.values():
            overlap = len(capsule_set & cluster.capsule_ids)
            if overlap >= max(3, len(cluster.capsule_ids) * 0.5):
                return cluster
        return None

    def _find_centroid(self, capsule_set: Set[str], capsules: Dict[str, Capsule]) -> Optional[str]:
        best_id, best_gravity = None, -1.0
        for cid in capsule_set:
            cap = capsules.get(cid)
            if cap and cap.gravity_score > best_gravity:
                best_gravity = cap.gravity_score
                best_id = cid
        return best_id

    def _update_cluster_edges(
        self, agreement_graph: Dict[str, Dict[str, float]], capsules: Dict[str, Capsule]
    ) -> None:
        self.cluster_edges.clear()
        cluster_list = list(self.clusters.values())
        for i, cl_a in enumerate(cluster_list):
            for cl_b in cluster_list[i + 1:]:
                scores: List[float] = []
                for cid_a in cl_a.capsule_ids:
                    if cid_a in agreement_graph:
                        for cid_b in cl_b.capsule_ids:
                            if cid_b in agreement_graph[cid_a]:
                                scores.append(agreement_graph[cid_a][cid_b])
                if scores:
                    avg_score = sum(scores) / len(scores)
                    self.cluster_edges[(cl_a.id, cl_b.id)] = avg_score
                    self.cluster_edges[(cl_b.id, cl_a.id)] = avg_score

    def activate_cluster(self, cluster_id: str, sustain: int = 5) -> List[str]:
        if cluster_id not in self.clusters:
            return []
        activated: List[str] = [cluster_id]
        self.clusters[cluster_id].fire(sustain)
        for (src_id, dst_id), strength in self.cluster_edges.items():
            if src_id == cluster_id and dst_id in self.clusters:
                neighbor = self.clusters[dst_id]
                cascade_amount = self.clusters[cluster_id].cascade_to(neighbor, strength)
                if cascade_amount >= neighbor.trigger_threshold:
                    neighbor.fire(sustain // 2)
                    activated.append(dst_id)
        return activated

    def decay_all(self) -> None:
        for cluster in self.clusters.values():
            cluster.decay()

    def get_active_clusters(self) -> List[Cluster]:
        return [c for c in self.clusters.values() if c.activation_level > 0.1]

    def get_ready_clusters(self) -> List[Cluster]:
        return [c for c in self.clusters.values() if 0.1 < c.activation_level < c.trigger_threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clusters": {cid: c.to_dict() for cid, c in self.clusters.items()},
            "edges": {f"{a}<->{b}": s for (a, b), s in self.cluster_edges.items()},
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.clusters = {}
        for cid, cdata in data.get("clusters", {}).items():
            self.clusters[cid] = Cluster.from_dict(cdata)
        self.cluster_edges = {}
        for key, strength in data.get("edges", {}).items():
            a, b = key.split("<->")
            self.cluster_edges[(a, b)] = strength


# ---------------------------------------------------------------------------
# 4.  Routing Engine
# ---------------------------------------------------------------------------

class RoutingEngine:
    """Deterministic routing-by-agreement with graduated thresholds."""

    def __init__(self, agreement_threshold: float = 0.63, routing_rounds: int = 4) -> None:
        self.agreement_threshold = agreement_threshold
        self.routing_rounds = routing_rounds
        self.agreement_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.routing_stats: Dict[str, Any] = {
            "cycles_run": 0, "merges_performed": 0,
            "total_agreements_computed": 0, "orbit0_merges": 0,
            "shadows_created": 0,
            "ergo_resolutions": 0, "tensions_created": 0,
            "tensions_compressed": 0, "tensions_resolved_stale": 0,
            "challenges_created": 0,
        }

    def route(
        self, capsules: Dict[str, Capsule],
        active_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Capsule], List[Tuple[str, str]], Set[str]]:
        if active_ids is None:
            active_ids = [
                cid for cid, c in capsules.items()
                if not c.is_shadow and not c.merged_into
                and c.activation_state in (ActivationState.ACTIVE, ActivationState.READY, ActivationState.COOLING)
            ]
        active_list = [capsules[cid] for cid in active_ids if cid in capsules]
        merges: List[Tuple[str, str]] = []
        shadows_created: Set[str] = set()

        for _round in range(self.routing_rounds):
            agreements: Dict[Tuple[str, str], float] = {}
            for i, cap_i in enumerate(active_list):
                if cap_i.is_shadow or cap_i.merged_into: continue
                for cap_j in active_list[i + 1:]:
                    if cap_j.is_shadow or cap_j.merged_into: continue
                    if cap_i.kind != cap_j.kind: continue
                    score = cap_i.evaluate_agreement(cap_j)
                    self.routing_stats["total_agreements_computed"] += 1
                    threshold = cap_i.get_merge_threshold(cap_j)
                    if score >= threshold:
                        agreements[(cap_i.id, cap_j.id)] = score
                        self.agreement_graph[cap_i.id][cap_j.id] = (
                            self.agreement_graph[cap_i.id].get(cap_j.id, 0) + score
                        ) / 2
                        self.agreement_graph[cap_j.id][cap_i.id] = (
                            self.agreement_graph[cap_j.id].get(cap_i.id, 0) + score
                        ) / 2

            sorted_agreements = sorted(agreements.items(), key=lambda x: x[1], reverse=True)
            merged_in_round: set = set()
            for (id_a, id_b), _score in sorted_agreements:
                if id_a in merged_in_round or id_b in merged_in_round: continue
                cap_a = capsules.get(id_a); cap_b = capsules.get(id_b)
                if not cap_a or not cap_b or cap_a.is_shadow or cap_b.is_shadow or cap_a.merged_into or cap_b.merged_into:
                    continue
                merged = cap_a.merge(cap_b)
                capsules[merged.id] = merged
                cap_a.is_shadow = True; cap_a.merged_into = merged.id
                cap_b.is_shadow = True; cap_b.merged_into = merged.id
                merges.append((id_a, id_b))
                shadows_created.add(id_a); shadows_created.add(id_b)
                merged_in_round.add(id_a); merged_in_round.add(id_b)
                self.routing_stats["merges_performed"] += 1
                self.routing_stats["shadows_created"] += 2
                if cap_a.orbit_level == 0 or cap_b.orbit_level == 0:
                    self.routing_stats["orbit0_merges"] += 1
                active_list = [c for c in active_list if c.id not in (id_a, id_b)]
                active_list.append(merged)
        self.routing_stats["cycles_run"] += 1
        return capsules, merges, shadows_created


# ---------------------------------------------------------------------------
# 5.  Gravity & Orbit Manager
# ---------------------------------------------------------------------------

class GravityOrbitManager:
    PROMOTE_THRESHOLD: float = 0.8
    DEMOTE_THRESHOLD: float = 0.2
    ORBIT0_DEMOTE_THRESHOLD: float = 0.1
    ORBIT0_DECAY_MULTIPLIER: float = 0.1

    def __init__(self, gravity_delta_success: float = 0.05,
                 gravity_decay_rate: float = 0.01, time_constant_days: float = 7.0) -> None:
        self.gravity_delta_success = gravity_delta_success
        self.gravity_decay_rate = gravity_decay_rate
        self.time_constant_days = time_constant_days

    def update_gravity(
        self, capsules: Dict[str, Capsule],
        active_ids: Optional[Sequence[str]] = None, current_frame: int = 0
    ) -> Dict[str, float]:
        if active_ids is None:
            active_ids = [cid for cid, c in capsules.items() if not c.is_shadow and not c.merged_into]
        results: Dict[str, float] = {}
        now = datetime.now()
        for cid in active_ids:
            cap = capsules.get(cid)
            if not cap or cap.is_shadow or cap.merged_into: continue
            days_since_use = (now - cap.last_used_at).total_seconds() / 86400.0
            decay_factor = math.exp(-days_since_use / max(1.0, self.time_constant_days))
            effective_decay = (
                self.gravity_decay_rate * self.ORBIT0_DECAY_MULTIPLIER
                if cap.orbit_level == 0 else self.gravity_decay_rate
            )
            cap.gravity_score *= (1.0 - effective_decay)
            cap.gravity_score *= (0.5 + 0.5 * decay_factor)
            cap.gravity_score = max(0.0, min(1.0, cap.gravity_score))

            if cap.orbit_level == 0:
                if cap.gravity_score < self.ORBIT0_DEMOTE_THRESHOLD:
                    cap.orbit_level = 1
            else:
                if cap.gravity_score >= self.PROMOTE_THRESHOLD and cap.orbit_level > 0:
                    cap.orbit_level = max(0, cap.orbit_level - 1)
                elif cap.gravity_score < self.DEMOTE_THRESHOLD and cap.orbit_level < 3:
                    cap.orbit_level = min(3, cap.orbit_level + 1)

            if cap.orbit_level == 0:
                cap.processing_lane = ProcessingLane.ERGO
            elif cap.orbit_level == 3:
                cap.processing_lane = ProcessingLane.INGESTION
            else:
                cap.processing_lane = ProcessingLane.COGNITIVE if cap.orbit_level == 1 else ProcessingLane.ARCHIVE

            bounds = LANE_BOUNDARIES.get(cap.orbit_level, (0.45, 0.72))
            cap.orbit_radius = (bounds[0] + bounds[1]) / 2.0

            if cap.orbit_level == 0 and cap.activation_state == ActivationState.COLD:
                cap.activation_state = ActivationState.READY
            elif cap.orbit_level == 3 and cap.gravity_score < 0.05:
                cap.activation_state = ActivationState.COLD

            results[cid] = cap.gravity_score
        return results

    def boost_gravity(self, capsule: Capsule, amount: Optional[float] = None, frame_number: int = 0) -> float:
        delta = amount if amount is not None else self.gravity_delta_success
        capsule.gravity_score = min(1.0, capsule.gravity_score + delta)
        capsule.touch(frame_number)
        return capsule.gravity_score


# ---------------------------------------------------------------------------
# 6.  Adaptive Resource Manager
# ---------------------------------------------------------------------------

class AdaptiveResourceManager:
    def __init__(self) -> None:
        self.cpu_count: int = self._detect_cpus()
        self.total_ram_mb: float = self._detect_ram_mb()
        self.max_cluster_budget: int = self._compute_max_clusters()
        self.current_cluster_budget: int = self.max_cluster_budget
        self.pressure_level: float = 0.0
        self.stats: Dict[str, Any] = {
            "cpu_count": self.cpu_count, "total_ram_mb": self.total_ram_mb,
            "max_cluster_budget": self.max_cluster_budget,
            "current_cluster_budget": self.current_cluster_budget,
            "budget_adjustments": 0,
        }

    @staticmethod
    def _detect_cpus() -> int:
        try: return os.cpu_count() or 4
        except: return 4

    @staticmethod
    def _detect_ram_mb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            return 4096.0

    def _compute_max_clusters(self) -> int:
        return min(self.cpu_count * 2, max(3, int(self.total_ram_mb / 1024)))

    def assess_pressure(self) -> float:
        try:
            import psutil
            return max(psutil.cpu_percent(interval=0.1) / 100.0, psutil.virtual_memory().percent / 100.0)
        except ImportError:
            return 0.3

    def adjust_budget(self, active_clusters: int) -> int:
        self.pressure_level = self.assess_pressure()
        if self.pressure_level > 0.8:
            self.current_cluster_budget = max(1, int(self.current_cluster_budget * 0.7))
        elif self.pressure_level > 0.6:
            self.current_cluster_budget = max(1, int(self.current_cluster_budget * 0.85))
        elif self.pressure_level < 0.3 and self.current_cluster_budget < self.max_cluster_budget:
            self.current_cluster_budget = min(self.max_cluster_budget, int(self.current_cluster_budget * 1.15))
        self.current_cluster_budget = min(self.current_cluster_budget, self.max_cluster_budget)
        self.stats["current_cluster_budget"] = self.current_cluster_budget
        self.stats["budget_adjustments"] += 1
        return self.current_cluster_budget

    def select_active_clusters(
        self, cluster_manager: ClusterManager, max_clusters: Optional[int] = None
    ) -> List[str]:
        budget = max_clusters if max_clusters is not None else self.current_cluster_budget
        active = cluster_manager.get_active_clusters()
        ready = cluster_manager.get_ready_clusters()
        selected = [c.id for c in active[:budget]]
        remaining = budget - len(selected)
        if remaining > 0:
            selected.extend(c.id for c in sorted(ready, key=lambda c: c.activation_level, reverse=True)[:remaining])
            remaining = budget - len(selected)
        if remaining > 0:
            cold = [c for c in cluster_manager.clusters.values() if c.activation_level <= 0.1]
            selected.extend(c.id for c in sorted(cold, key=lambda c: c.fire_count, reverse=True)[:remaining])
        return selected[:budget]


# ---------------------------------------------------------------------------
# 7.  Temporal Graph / FrameNode
# ---------------------------------------------------------------------------

@dataclass
class FrameNode:
    frame_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    active_cluster_ids: List[str] = field(default_factory=list)
    active_capsule_ids: List[str] = field(default_factory=list)
    active_count: int = 0
    cluster_count: int = 0
    orbit_distribution: Dict[int, int] = field(default_factory=dict)
    activation_distribution: Dict[str, int] = field(default_factory=dict)
    max_gravity: float = 0.0
    merges_this_cycle: int = 0
    shadows_created: int = 0
    tensions_created: int = 0
    tensions_resolved: int = 0
    ergo_resolutions: int = 0
    reprocessed_count: int = 0
    challenges_created: int = 0
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameNode":
        data = dict(data)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class TimelineStore:
    def __init__(self, store_path: Path = DEFAULT_TEMPORAL_STORE_PATH) -> None:
        self.store_path = store_path
        self.frames: List[FrameNode] = []

    def append(self, frame: FrameNode) -> None:
        self.frames.append(frame)
        with open(self.store_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(frame.to_dict(), default=str) + "\n")

    def load(self, max_frames: int = 2000) -> None:
        if not self.store_path.exists(): return
        loaded: List[FrameNode] = []
        with open(self.store_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try: loaded.append(FrameNode.from_dict(json.loads(line)))
                except: pass
        self.frames = loaded[-max_frames:]

    def replay(self, start_index: int = 0, end_index: Optional[int] = None) -> List[Dict[str, Any]]:
        end = end_index if end_index is not None else len(self.frames)
        return [f.to_dict() for f in self.frames[start_index:end]]

    def stats(self) -> Dict[str, Any]:
        if not self.frames: return {"total_frames": 0}
        orbit_counts: Dict[int, int] = defaultdict(int)
        for f in self.frames:
            for level, count in f.orbit_distribution.items():
                orbit_counts[level] += count
        return {
            "total_frames": len(self.frames),
            "avg_active_capsules": sum(f.active_count for f in self.frames) / len(self.frames),
            "avg_active_clusters": sum(f.cluster_count for f in self.frames) / len(self.frames),
            "total_merges": sum(f.merges_this_cycle for f in self.frames),
            "total_shadows": sum(f.shadows_created for f in self.frames),
            "total_tensions": sum(f.tensions_created for f in self.frames),
            "total_ergo_resolutions": sum(f.ergo_resolutions for f in self.frames),
            "total_reprocessed": sum(f.reprocessed_count for f in self.frames),
            "total_challenges": sum(f.challenges_created for f in self.frames),
            "orbit_totals": dict(orbit_counts),
        }


# ---------------------------------------------------------------------------
# 8.  Persistent Store
# ---------------------------------------------------------------------------

class PersistentStore:
    def __init__(self, store_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH) -> None:
        self.store_path = store_path

    def save(self, capsules: Dict[str, Capsule], clusters: Optional[Dict[str, Any]] = None,
             extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "capsules": {cid: c.to_dict_safe() for cid, c in capsules.items()},
            "shadow_count": sum(1 for c in capsules.values() if c.is_shadow),
            "active_count": sum(1 for c in capsules.values() if not c.is_shadow and not c.merged_into),
            "tension_count": sum(1 for c in capsules.values() if c.kind == CapsuleKind.TENSION),
            "challenge_count": sum(1 for c in capsules.values() if c.kind == CapsuleKind.CHALLENGE),
            "reprocess_exhausted": sum(1 for c in capsules.values() if c.reprocess_exhausted),
            "orbit0_count": sum(1 for c in capsules.values() if c.orbit_level == 0 and not c.is_shadow),
            "saved_at": datetime.now().isoformat(),
        }
        if clusters: payload["clusters"] = clusters
        if extra: payload.update(extra)
        with open(self.store_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def load(self) -> Tuple[Dict[str, Capsule], Optional[Dict[str, Any]]]:
        if not self.store_path.exists(): return {}, None
        with open(self.store_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        capsules: Dict[str, Capsule] = {}
        for cid, cdata in payload.get("capsules", {}).items():
            capsules[cid] = Capsule.from_dict(cdata)
        return capsules, payload.get("clusters")


# ---------------------------------------------------------------------------
# 9.  Embedding helpers
# ---------------------------------------------------------------------------

def deterministic_embed(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{2,}", text.lower())
    if not tokens: return vec
    for i, token in enumerate(tokens):
        h = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(h[:4], "big") % dim
        vec[bucket] += 1.0 + (0.35 if i >= len(tokens) * 0.8 else 0.0)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9: vec /= norm
    return vec


# ---------------------------------------------------------------------------
# 10. Seed Capsule Initializer
# ---------------------------------------------------------------------------

def initialize_seed_capsules() -> Dict[str, Capsule]:
    seeds: Dict[str, Capsule] = {}

    nucleus_capsules = [
        ("nucleus_core_identity", "Core Identity",
         {"traits": ["deterministic", "structured", "analytical", "curious", "self-correcting"],
          "purpose": "Understand through routing-by-agreement with rotational feedback",
          "invariants": ["no_gradient_descent", "capsule_primacy", "agreement_over_prediction",
                        "ergo_reconciliation", "core_inertia"]},
         ["kerca", "capsule", "identity", "kernel", "deterministic", "ergoregion"],
         "Nucleus — core cognitive identity. KERCA rotating consensus engine."),
        ("nucleus_routing_expert", "Routing Expert",
         {"specialties": ["agreement", "routing", "consensus", "merge", "reconciliation"],
          "routing_philosophy": "Agreement is structural. Disagreement is productive. Contradictions are data."},
         ["routing", "agreement", "merge", "threshold", "capsule", "tension"],
         "Nucleus — routing-by-agreement with ergoregion reconciliation."),
        ("nucleus_orbit_keeper", "Orbit Keeper",
         {"domain": "orbital_mechanics",
          "responsibility": "Maintain gravitational hierarchy with core inertia protection"},
         ["orbit", "gravity", "lane", "salience", "decay", "hierarchy", "inertia"],
         "Nucleus — orbital hierarchy with Kerr rotational dynamics."),
        ("nucleus_ergo_overseer", "Ergoregion Overseer",
         {"domain": "reconciliation",
          "responsibility": "Force near-miss resolution, manage tension lifecycle, prevent feedback explosion"},
         ["ergoregion", "reconciliation", "tension", "resolution", "feedback", "reprocessing"],
         "Nucleus — manages forced-reconciliation zone with budget enforcement."),
    ]

    for cid, name, state, keywords, description in nucleus_capsules:
        emb = deterministic_embed(name + " " + " ".join(keywords))
        seeds[cid] = Capsule(
            id=cid, kind=CapsuleKind.NUCLEUS, name=name, state=state,
            content={"keywords": keywords, "description": description},
            embedding=emb, orbit_level=0, orbit_radius=0.08, gravity_score=1.0,
            agreement_score=1.0, confidence=1.0, activation_state=ActivationState.READY,
            processing_lane=ProcessingLane.ERGO, ergo_processed=True,
            reprocess_exhausted=True,  # Core capsules never reprocessed
            core_inertia=10,            # Highly stable from start
        )

    domain_seeds = [
        ("topic_capsule_networks", "Capsule Networks",
         ["capsule", "routing", "agreement", "hinton", "pose", "transformation"],
         "Capsule network theory: dynamic routing, pose matrices, part-whole hierarchies."),
        ("topic_kerca_architecture", "KERCA Architecture",
         ["kerca", "capsule", "orbital", "gravity", "routing", "ergoregion", "rotation", "inertia"],
         "Kerr Engine for Routing and Capsule Agreement: rotating consensus with ergoregion."),
        ("topic_deterministic_systems", "Deterministic Systems",
         ["deterministic", "algorithm", "consensus", "hash", "embedding"],
         "Deterministic algorithms — no gradient descent, no neural networks."),
        ("topic_feedback_systems", "Feedback Systems",
         ["feedback", "loop", "recursive", "reprocessing", "reconciliation", "temporal"],
         "Rotational feedback with bounded reprocessing budgets."),
        ("skill_text_analysis", "Text Analysis",
         ["text", "analysis", "parse", "extract", "keyword", "summarize"],
         "Skill: Extract structured information from unstructured text."),
        ("skill_cross_reference", "Cross Referencing",
         ["cross", "reference", "compare", "contradiction", "consensus", "source"],
         "Skill: Cross-reference claims across multiple sources."),
        ("memory_research_log", "Research Log",
         ["research", "log", "finding", "insight", "discovery", "pattern"],
         "Memory: Tracks research findings and emergent patterns."),
        ("workflow_research_synthesis", "Research Synthesis",
         ["research", "synthesis", "review", "integrate", "summarize", "hypothesis"],
         "Workflow: Synthesize findings into coherent hypotheses."),
    ]

    for cid, name, keywords, description in domain_seeds:
        emb = deterministic_embed(name + " " + " ".join(keywords))
        seeds[cid] = Capsule(
            id=cid,
            kind=CapsuleKind.TOPIC if cid.startswith("topic_") else (
                CapsuleKind.SKILL if cid.startswith("skill_") else (
                    CapsuleKind.MEMORY if cid.startswith("memory_") else CapsuleKind.WORKFLOW)),
            name=name, content={"keywords": keywords, "description": description},
            embedding=emb, orbit_level=2, orbit_radius=0.55, gravity_score=0.35,
            activation_state=ActivationState.READY, processing_lane=ProcessingLane.ARCHIVE,
        )

    return seeds


# ---------------------------------------------------------------------------
# 11. KERCA Kernel — with all risk mitigations integrated
# ---------------------------------------------------------------------------

class KERCAKernel:
    """KERCA: rotating consensus engine with ergoregion and full risk mitigations.

    Risk Mitigations (integrated, not bolted on):
      RISK 1 — Feedback explosion:
        • reprocess_count / max_reprocess (default 3)
        • reprocess_decay factor on re-entry strength
        • Orbit 0 capsules skip reprocessing entirely
      RISK 2 — Tension proliferation:
        • Significance gate: gravity ≥ 0.25 to create tension
        • compress_tensions(): merge tensions about same pair
        • resolve_stale_tensions(): remove tensions where sources are gone
      RISK 3 — Orbit 0 contamination:
        • Core inertia: challenges must accumulate across cycles
        • External challenge capsules instead of direct modification
        • orbit0_modification_protocol(): multi-cycle consensus required
    """

    def __init__(self, load_existing: bool = True,
                 store_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
                 timeline_path: Path = DEFAULT_TEMPORAL_STORE_PATH) -> None:
        self.store = PersistentStore(store_path)
        self.timeline = TimelineStore(timeline_path)
        self.router = RoutingEngine(agreement_threshold=0.63, routing_rounds=4)
        self.gravity_orbit = GravityOrbitManager()
        self.resource_mgr = AdaptiveResourceManager()
        self.cluster_mgr = ClusterManager()

        if load_existing:
            self.capsules, cluster_data = self.store.load()
        else:
            self.capsules: Dict[str, Capsule] = {}
            cluster_data = None
        if not self.capsules:
            self.capsules = initialize_seed_capsules()
            self.store.save(self.capsules)
        if cluster_data:
            self.cluster_mgr.from_dict(cluster_data)

        self.timeline.load()
        self.cycle_count: int = 0
        self.running: bool = False
        self._input_queue: List[Dict[str, Any]] = []

    # ==================================================================
    #  PUBLIC API
    # ==================================================================

    def ingest_input(self, text: str, source: str = "user", activate: bool = True) -> List[str]:
        chunks = self._chunk_text(text)
        created_ids: List[str] = []
        for i, chunk in enumerate(chunks):
            cid = f"ingest_{self.cycle_count}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            emb = deterministic_embed(chunk)
            cap = Capsule(
                id=cid, kind=CapsuleKind.TOPIC, name=f"Ingest::{source}::{i+1}",
                state={"raw_text": chunk[:500], "source": source},
                content={"keywords": self._extract_keywords(chunk)[:30], "text": chunk,
                        "description": chunk[:200]},
                embedding=emb, orbit_level=3, orbit_radius=0.85, gravity_score=0.15,
                activation_state=ActivationState.READY, processing_lane=ProcessingLane.INGESTION,
                sustain_cycles=3 if activate else 0,
            )
            self.capsules[cid] = cap
            created_ids.append(cid)
        self._input_queue.append({"text": text, "source": source, "chunk_count": len(chunks)})
        return created_ids

    def activate_by_query(self, query_text: str) -> List[str]:
        query_emb = deterministic_embed(query_text)
        scores: List[Tuple[str, float]] = []
        for cid, cap in self.capsules.items():
            if cap.is_shadow or cap.merged_into: continue
            if cap.embedding is not None:
                nq = float(np.linalg.norm(query_emb))
                nc = float(np.linalg.norm(cap.embedding))
                scores.append((cid, float(np.dot(query_emb, cap.embedding) / max(1e-9, nq * nc))))
            else:
                scores.append((cid, 0.1))
        scores.sort(key=lambda x: x[1], reverse=True)

        activated: Set[str] = set()
        for cid, score in scores[:10]:
            if score < 0.2: continue
            cap = self.capsules.get(cid)
            if cap and cap.cluster_id:
                cascaded = self.cluster_mgr.activate_cluster(cap.cluster_id, sustain=6)
                activated.update(cascaded)
        for cluster_id in activated:
            cluster = self.cluster_mgr.clusters.get(cluster_id)
            if cluster:
                for cid in cluster.capsule_ids:
                    cap = self.capsules.get(cid)
                    if cap and not cap.is_shadow and not cap.merged_into:
                        cap.activation_state = ActivationState.ACTIVE
                        cap.sustain_cycles = cluster.sustain_cycles
                        cap.touch(self.cycle_count)
        return list(activated)

    def run_cycle(self, trigger_clusters: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main cognition cycle with all KERCA protections."""
        self.cycle_count += 1
        t0 = time.time()
        tensions_created = 0
        ergo_resolutions = 0
        reprocessed = 0
        challenges_created = 0
        tensions_resolved = 0

        # --- Trigger clusters ---
        if trigger_clusters:
            for cid in trigger_clusters:
                if cid in self.cluster_mgr.clusters:
                    self.cluster_mgr.activate_cluster(cid, sustain=5)

        # --- Sync activation states ---
        self._sync_activation_states()

        # --- Cluster formation (periodic) ---
        if self.cycle_count % 3 == 0:
            self.cluster_mgr.form_clusters(self.capsules, self.router.agreement_graph)

        # --- Select active clusters ---
        active_cluster_ids = self.resource_mgr.select_active_clusters(self.cluster_mgr)
        self.resource_mgr.adjust_budget(len(active_cluster_ids))

        # --- Gather active capsules ---
        active_ids = self._get_capsules_in_clusters(active_cluster_ids)
        for cid, cap in self.capsules.items():
            if cap.is_active() and cid not in active_ids:
                active_ids.append(cid)
            if cap.kind in (CapsuleKind.TENSION, CapsuleKind.CHALLENGE) and cid not in active_ids:
                active_ids.append(cid)
            if cap.needs_review and cid not in active_ids:
                active_ids.append(cid)

        # --- Route ---
        self.capsules, merges, shadows_created = self.router.route(self.capsules, active_ids)
        self.gravity_orbit.update_gravity(self.capsules, active_ids, self.cycle_count)

        # --- Boost gravity ---
        for cid in active_ids:
            cap = self.capsules.get(cid)
            if cap and not cap.is_shadow and not cap.merged_into:
                self.gravity_orbit.boost_gravity(cap, frame_number=self.cycle_count)

        # --- Decay clusters ---
        self.cluster_mgr.decay_all()

        # --- Decay sustain counters ---
        for cap in self.capsules.values():
            if cap.sustain_cycles > 0:
                cap.sustain_cycles -= 1
            if cap.sustain_cycles <= 0 and cap.activation_state == ActivationState.ACTIVE:
                cap.activation_state = ActivationState.COOLING
            elif cap.sustain_cycles <= 0 and cap.activation_state == ActivationState.COOLING:
                cap.activation_state = ActivationState.READY

        # ================================================================
        #  KERCA PROTECTIONS (RISK 1-3)
        # ================================================================

        # --- RISK 1+2: Ergoregion reconciliation (every 5 cycles) ---
        if self.cycle_count % 5 == 0 and self.cycle_count > 0:
            ergo = self.reconciliation_cycle()
            ergo_resolutions = ergo["resolutions"]
            tensions_created = ergo["tensions_created"]

        # --- RISK 1: Temporal reprocessing with budget (every 12 cycles) ---
        if self.cycle_count % 12 == 0 and self.cycle_count > 0:
            tr = self.replay_reprocess(lookback_frames=8)
            reprocessed = tr["reprocessed"]

        # --- RISK 2: Tension compression (every 10 cycles) ---
        if self.cycle_count % 10 == 0 and self.cycle_count > 0:
            compressed = self.compress_tensions()
            resolved = self.resolve_stale_tensions()
            tensions_resolved = compressed + resolved
            self.router.routing_stats["tensions_compressed"] += compressed
            self.router.routing_stats["tensions_resolved_stale"] += resolved

        # --- RISK 3: Orbit 0 challenge evaluation (every 20 cycles) ---
        if self.cycle_count % 20 == 0 and self.cycle_count > 0:
            challenges_created = self.evaluate_orbit0_challenges()

        # ================================================================

        # --- Build frame ---
        orbit_dist: Dict[int, int] = defaultdict(int)
        activation_dist: Dict[str, int] = defaultdict(int)
        max_g = 0.0
        for cid in active_ids:
            cap = self.capsules.get(cid)
            if cap and not cap.is_shadow and not cap.merged_into:
                orbit_dist[cap.orbit_level] += 1
                activation_dist[cap.activation_state.value] += 1
                max_g = max(max_g, cap.gravity_score)

        frame = FrameNode(
            timestamp=datetime.now(), frame_number=self.cycle_count,
            active_cluster_ids=active_cluster_ids, active_capsule_ids=active_ids,
            active_count=len(active_ids), cluster_count=len(active_cluster_ids),
            orbit_distribution=dict(orbit_dist), activation_distribution=dict(activation_dist),
            max_gravity=round(max_g, 4), merges_this_cycle=len(merges),
            shadows_created=len(shadows_created),
            tensions_created=tensions_created, tensions_resolved=tensions_resolved,
            ergo_resolutions=ergo_resolutions, reprocessed_count=reprocessed,
            challenges_created=challenges_created, edges=merges,
            metadata={"cycle": self.cycle_count, "pressure": self.resource_mgr.pressure_level,
                      "cluster_budget": self.resource_mgr.current_cluster_budget},
        )
        self.timeline.append(frame)
        self.store.save(self.capsules, clusters=self.cluster_mgr.to_dict(),
                       extra={"cycle_count": self.cycle_count,
                              "routing_stats": self.router.routing_stats,
                              "timeline_stats": self.timeline.stats()})

        return {
            "cycle": self.cycle_count,
            "active_capsules": len(active_ids),
            "active_clusters": len(active_cluster_ids),
            "merges_this_cycle": len(merges),
            "shadows_created": len(shadows_created),
            "tensions_created": tensions_created,
            "tensions_resolved": tensions_resolved,
            "ergo_resolutions": ergo_resolutions,
            "reprocessed": reprocessed,
            "challenges_created": challenges_created,
            "orbit_distribution": dict(orbit_dist),
            "max_gravity": round(max_g, 4),
            "total_capsules": len(self.capsules),
            "active_capsules_total": sum(1 for c in self.capsules.values() if c.is_active()),
            "shadows_total": sum(1 for c in self.capsules.values() if c.is_shadow),
            "ready_capsules": sum(1 for c in self.capsules.values() if c.activation_state == ActivationState.READY),
            "tension_capsules": sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.TENSION),
            "challenge_capsules": sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.CHALLENGE),
            "reprocess_exhausted": sum(1 for c in self.capsules.values() if c.reprocess_exhausted),
            "orbit0_count": sum(1 for c in self.capsules.values() if c.orbit_level == 0 and not c.is_shadow),
            "clusters_total": len(self.cluster_mgr.clusters),
            "elapsed_seconds": round(time.time() - t0, 4),
            "pressure": round(self.resource_mgr.pressure_level, 2),
            "cluster_budget": self.resource_mgr.current_cluster_budget,
        }

    # ==================================================================
    #  KERCA: RISK 1+2 — Ergoregion Reconciliation
    # ==================================================================

    def reconciliation_cycle(self) -> Dict[str, int]:
        """Force resolution of near-miss agreements.

        RISK 2 mitigation: only create tensions for capsules with gravity ≥ 0.25.
        """
        near_misses = []
        for cid_a, neighbors in self.router.agreement_graph.items():
            cap_a = self.capsules.get(cid_a)
            if not cap_a or cap_a.is_shadow or cap_a.merged_into: continue
            for cid_b, score in neighbors.items():
                cap_b = self.capsules.get(cid_b)
                if not cap_b or cap_b.is_shadow or cap_b.merged_into: continue

                # RISK 2: Significance gate — low gravity capsules don't create tensions
                if cap_a.gravity_score < 0.25 or cap_b.gravity_score < 0.25:
                    continue

                pair_threshold = cap_a.get_merge_threshold(cap_b)
                if pair_threshold * 0.85 <= score < pair_threshold:
                    near_misses.append((cid_a, cid_b, score, pair_threshold))

        near_misses.sort(key=lambda x: x[3] - x[2])
        resolutions = 0
        tensions_created = 0

        for cid_a, cid_b, score, threshold in near_misses[:30]:
            cap_a = self.capsules.get(cid_a)
            cap_b = self.capsules.get(cid_b)
            if not cap_a or not cap_b: continue

            # Expanded context re-evaluation
            expanded_a = cap_a.content.get("keywords", []) + cap_b.content.get("keywords", [])
            expanded_b = cap_b.content.get("keywords", []) + cap_a.content.get("keywords", [])
            shared_score = len(set(expanded_a) & set(expanded_b)) / max(1, len(set(expanded_a) | set(expanded_b)))

            if shared_score >= threshold:
                merged = cap_a.merge(cap_b)
                self.capsules[merged.id] = merged
                cap_a.is_shadow = True; cap_a.merged_into = merged.id
                cap_b.is_shadow = True; cap_b.merged_into = merged.id
                merged.ergo_processed = True
                self.router.routing_stats["merges_performed"] += 1
                self.router.routing_stats["shadows_created"] += 2
                self.router.routing_stats["ergo_resolutions"] += 1
                resolutions += 1
                continue

            # Create tension capsule (only if both capsules pass significance gate)
            tension_id = f"tension_{hashlib.md5(f'{cid_a}+{cid_b}'.encode()).hexdigest()[:10]}"
            tension = Capsule(
                id=tension_id, kind=CapsuleKind.TENSION,
                name=f"Tension: {cap_a.name[:25]} vs {cap_b.name[:25]}",
                state={"capsule_a": cid_a, "capsule_b": cid_b,
                       "agreement_score": score, "threshold": threshold,
                       "gap": round(threshold - score, 4), "unresolved": True},
                content={
                    "keywords": list(set(cap_a.content.get("keywords", []) + cap_b.content.get("keywords", []))),
                    "description": f"Unresolved: '{cap_a.name}' vs '{cap_b.name}'. "
                                  f"Score {score:.3f} < {threshold:.3f}.",
                    "perspectives": [cap_a.content.get("description", "")[:200],
                                    cap_b.content.get("description", "")[:200]],
                },
                embedding=(cap_a.embedding + cap_b.embedding) / 2 if (
                    cap_a.embedding is not None and cap_b.embedding is not None
                ) else cap_a.embedding,
                orbit_level=1, gravity_score=(cap_a.gravity_score + cap_b.gravity_score) / 2,
                activation_state=ActivationState.READY, processing_lane=ProcessingLane.ERGO,
            )
            self.capsules[tension_id] = tension
            cap_a.tension_pairs.append(cid_b)
            cap_b.tension_pairs.append(cid_a)
            tensions_created += 1

        self.router.routing_stats["tensions_created"] += tensions_created
        return {"resolutions": resolutions, "tensions_created": tensions_created}

    # ==================================================================
    #  KERCA: RISK 1 — Temporal Reprocessing (with budget)
    # ==================================================================

    def replay_reprocess(self, lookback_frames: int = 5) -> Dict[str, Any]:
        """Re-evaluate past capsules against current state.

        RISK 1 mitigation:
          • Skip capsules with exhausted reprocessing budget
          • Apply decay factor to re-entry strength
          • Skip Orbit 0 capsules entirely
        """
        if len(self.timeline.frames) < lookback_frames:
            return {"reprocessed": 0, "description": "Not enough history"}

        old_frame = self.timeline.frames[-lookback_frames]
        reprocessed = 0

        for old_cid in old_frame.active_capsule_ids:
            old_cap = self.capsules.get(old_cid)
            if not old_cap or old_cap.is_shadow or old_cap.merged_into:
                continue

            # RISK 1: Check reprocessing budget
            if not old_cap.can_reprocess():
                old_cap.reprocess_exhausted = True
                old_cap.needs_review = False
                continue

            # RISK 1: Apply decay factor
            budget_factor = 1.0 - (old_cap.reprocess_count * old_cap.reprocess_decay)
            budget_factor = max(0.3, budget_factor)  # Minimum 30% strength

            for cid, current_cap in self.capsules.items():
                if current_cap.is_shadow or current_cap.merged_into: continue
                if cid == old_cid: continue

                raw_score = old_cap.evaluate_agreement(current_cap)
                score = raw_score * budget_factor  # RISK 1: dampened re-entry
                threshold = old_cap.get_merge_threshold(current_cap)

                if score >= threshold:
                    old_cap.gravity_score = min(1.0, old_cap.gravity_score + 0.02)
                    reprocessed += 1
                elif score < threshold * 0.5 and current_cap.orbit_level == 0:
                    # RISK 3: Don't modify orbit 0. Create external challenge.
                    self._create_challenge(old_cap, current_cap, raw_score, threshold)
                    reprocessed += 1
                elif score < threshold * 0.5:
                    old_cap.needs_review = True
                    reprocessed += 1

            old_cap.reprocess_count += 1

            # RISK 1: Mark exhausted if budget spent
            if old_cap.reprocess_count >= old_cap.max_reprocess:
                old_cap.reprocess_exhausted = True
                old_cap.needs_review = False
                old_cap.metadata["reprocess_exhausted_at"] = datetime.now().isoformat()

        return {"reprocessed": reprocessed, "frame_revisited": old_frame.frame_number,
                "description": "Temporal reprocessing complete (budget-enforced)"}

    # ==================================================================
    #  KERCA: RISK 2 — Tension Lifecycle Management
    # ==================================================================

    def compress_tensions(self) -> int:
        """Merge tension capsules that reference the same source pair."""
        tensions: Dict[str, Capsule] = {
            cid: c for cid, c in self.capsules.items()
            if c.kind == CapsuleKind.TENSION and not c.is_shadow and not c.merged_into
        }
        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for cid, cap in tensions.items():
            a = cap.state.get("capsule_a", "")
            b = cap.state.get("capsule_b", "")
            groups[tuple(sorted([a, b]))].append(cid)

        merged = 0
        for pair, tids in groups.items():
            if len(tids) < 2: continue
            base = self.capsules[tids[0]]
            for tid in tids[1:]:
                other = self.capsules.get(tid)
                if not other or other.is_shadow: continue
                result = base.merge(other)
                self.capsules[result.id] = result
                base.is_shadow = True; base.merged_into = result.id
                other.is_shadow = True; other.merged_into = result.id
                base = result
                merged += 1
        return merged

    def resolve_stale_tensions(self) -> int:
        """Remove tensions whose source capsules no longer exist."""
        resolved = 0
        for cid, cap in list(self.capsules.items()):
            if cap.kind != CapsuleKind.TENSION: continue
            a = self.capsules.get(cap.state.get("capsule_a", ""))
            b = self.capsules.get(cap.state.get("capsule_b", ""))
            if not a or a.is_shadow or not b or b.is_shadow:
                cap.is_shadow = True
                cap.merged_into = "resolved_stale"
                cap.metadata["resolution"] = "sources_gone"
                resolved += 1
            elif a.merged_into and a.merged_into == b.merged_into:
                cap.is_shadow = True
                cap.merged_into = a.merged_into
                cap.metadata["resolution"] = "sources_merged"
                resolved += 1
        return resolved

    # ==================================================================
    #  KERCA: RISK 3 — Orbit 0 Protection
    # ==================================================================

    def _create_challenge(self, source: Capsule, target: Capsule, score: float, threshold: float) -> None:
        """Create an external challenge capsule rather than modifying orbit 0 directly."""
        challenge_id = f"challenge_{hashlib.md5(f'{source.id}+{target.id}'.encode()).hexdigest()[:10]}"
        challenge = Capsule(
            id=challenge_id, kind=CapsuleKind.CHALLENGE,
            name=f"Challenge: {source.name[:20]} → {target.name[:20]}",
            state={"source": source.id, "target": target.id,
                   "score": score, "threshold": threshold, "gap": round(threshold - score, 4)},
            content={"description": f"Temporal reprocessing found disagreement: "
                                   f"'{source.name}' challenges core capsule '{target.name}'."},
            orbit_level=1, gravity_score=0.35,
            activation_state=ActivationState.READY, processing_lane=ProcessingLane.ERGO,
        )
        self.capsules[challenge_id] = challenge
        target.core_inertia += 1
        target.core_agreements.append(challenge_id)
        self.router.routing_stats["challenges_created"] += 1

    def evaluate_orbit0_challenges(self) -> int:
        """Process accumulated challenges against orbit 0 capsules.

        RISK 3: Core inertia constraint.
        Only modify orbit 0 if challenges meet the multi-cycle consensus threshold.
        """
        challenges_processed = 0
        for cid, cap in list(self.capsules.items()):
            if cap.orbit_level != 0 or cap.is_shadow: continue

            # Count unresolved challenges directed at this capsule
            active_challenges = [
                c for cid_c, c in self.capsules.items()
                if c.kind == CapsuleKind.CHALLENGE
                and not c.is_shadow
                and c.state.get("target") == cid
            ]

            if len(active_challenges) >= cap.core_consensus_required:
                # Threshold met: record the challenge consensus
                cap.core_modification_count += 1
                cap.core_inertia = 0
                cap.metadata[f"challenge_consensus_{cap.core_modification_count}"] = {
                    "time": datetime.now().isoformat(),
                    "challenge_count": len(active_challenges),
                    "challenge_ids": [c.id for c in active_challenges],
                }
                # Mark challenges as resolved
                for challenge in active_challenges:
                    challenge.is_shadow = True
                    challenge.merged_into = cid
                    challenge.metadata["resolution"] = "consensus_reached"
                challenges_processed += len(active_challenges)

        return challenges_processed

    # ==================================================================
    #  HELPERS
    # ==================================================================

    def _sync_activation_states(self) -> None:
        for cluster in self.cluster_mgr.clusters.values():
            is_active = cluster.activation_level >= cluster.trigger_threshold
            for cid in cluster.capsule_ids:
                cap = self.capsules.get(cid)
                if cap and not cap.is_shadow and not cap.merged_into:
                    if is_active and cap.activation_state != ActivationState.ACTIVE:
                        cap.activation_state = ActivationState.ACTIVE
                        cap.sustain_cycles = cluster.sustain_cycles

    def _get_capsules_in_clusters(self, cluster_ids: List[str]) -> List[str]:
        result: List[str] = []
        for cid in cluster_ids:
            cluster = self.cluster_mgr.clusters.get(cid)
            if cluster:
                for cap_id in cluster.capsule_ids:
                    cap = self.capsules.get(cap_id)
                    if cap and not cap.is_shadow and not cap.merged_into:
                        result.append(cap_id)
        return result

    def query(self, text: str, top_k: int = 5, activate: bool = True) -> List[Dict[str, Any]]:
        if activate:
            self.activate_by_query(text)
        query_emb = deterministic_embed(text)
        scored: List[Tuple[float, Capsule]] = []
        for cap in self.capsules.values():
            if cap.is_shadow or cap.merged_into: continue
            if cap.embedding is not None:
                nq = float(np.linalg.norm(query_emb))
                nc = float(np.linalg.norm(cap.embedding))
                score = float(np.dot(query_emb, cap.embedding) / max(1e-9, nq * nc))
            else:
                score = 0.1
            scored.append((score, cap))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{
            "id": c.id, "name": c.name, "kind": c.kind.value,
            "orbit_level": c.orbit_level, "orbit_label": ORBIT_LABELS.get(c.orbit_level, "?"),
            "gravity": round(c.gravity_score, 4), "score": round(score, 4),
            "activation": c.activation_state.value, "cluster_id": c.cluster_id,
            "tension": len(c.tension_pairs) > 0, "needs_review": c.needs_review,
            "reprocess_exhausted": c.reprocess_exhausted,
            "core_inertia": c.core_inertia if c.orbit_level == 0 else 0,
            "content_summary": str(c.content.get("description", c.content.get("text", ""))[:120]),
        } for score, c in scored[:top_k]]

    def status(self) -> Dict[str, Any]:
        orbit_counts: Dict[int, int] = defaultdict(int)
        kind_counts: Dict[str, int] = defaultdict(int)
        activation_counts: Dict[str, int] = defaultdict(int)
        for cap in self.capsules.values():
            if not cap.is_shadow and not cap.merged_into:
                orbit_counts[cap.orbit_level] += 1
                kind_counts[cap.kind.value] += 1
                activation_counts[cap.activation_state.value] += 1
        return {
            "total_capsules": len(self.capsules),
            "active_capsules": sum(1 for c in self.capsules.values() if c.is_active()),
            "ready_capsules": sum(1 for c in self.capsules.values() if c.activation_state == ActivationState.READY),
            "cooling_capsules": sum(1 for c in self.capsules.values() if c.activation_state == ActivationState.COOLING),
            "cold_capsules": sum(1 for c in self.capsules.values() if c.activation_state == ActivationState.COLD),
            "shadows": sum(1 for c in self.capsules.values() if c.is_shadow),
            "tensions": sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.TENSION),
            "challenges": sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.CHALLENGE),
            "needs_review": sum(1 for c in self.capsules.values() if c.needs_review),
            "reprocess_exhausted": sum(1 for c in self.capsules.values() if c.reprocess_exhausted),
            "orbit0_count": sum(1 for c in self.capsules.values() if c.orbit_level == 0 and not c.is_shadow),
            "orbit_distribution": dict(orbit_counts),
            "kind_distribution": dict(kind_counts),
            "activation_distribution": dict(activation_counts),
            "total_clusters": len(self.cluster_mgr.clusters),
            "active_clusters": len(self.cluster_mgr.get_active_clusters()),
            "ready_clusters": len(self.cluster_mgr.get_ready_clusters()),
            "cycle_count": self.cycle_count,
            "routing_stats": dict(self.router.routing_stats),
            "timeline_stats": self.timeline.stats(),
            "resource_stats": dict(self.resource_mgr.stats),
            "pressure": round(self.resource_mgr.pressure_level, 2),
            "cluster_budget": self.resource_mgr.current_cluster_budget,
        }

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 250) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            if end >= len(text): break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _extract_keywords(text: str, min_len: int = 3, max_keywords: int = 40) -> List[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{2,}", text.lower())
        freq: Dict[str, int] = defaultdict(int)
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "are", "has", "was", "not", "but"}
        for t in tokens:
            if len(t) >= min_len and t not in stopwords:
                freq[t] += 1
        return sorted(freq, key=lambda k: freq[k], reverse=True)[:max_keywords]


# ---------------------------------------------------------------------------
# 12.  Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_kernel(load_existing: bool = True,
                     store_path: Optional[Path] = None,
                     timeline_path: Optional[Path] = None) -> KERCAKernel:
    sp = store_path or DEFAULT_KNOWLEDGE_BASE_PATH
    tp = timeline_path or DEFAULT_TEMPORAL_STORE_PATH
    kernel = KERCAKernel(load_existing=load_existing, store_path=sp, timeline_path=tp)
    kernel.resource_mgr.adjust_budget(0)
    return kernel


# ---------------------------------------------------------------------------
# 13.  CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("KERCA Kernel v1.0 — Smoke Test")
    print("=" * 60)
    kernel = bootstrap_kernel(load_existing=False)
    s = kernel.status()
    print(f"Capsules: {s['total_capsules']}")
    print(f"Orbits: {s['orbit_distribution']}")
    print(f"Orbit 0: {s['orbit0_count']}")
    print(f"Tensions: {s['tensions']}")
    print(f"Challenges: {s['challenges']}")
    print(f"Max cluster budget: {kernel.resource_mgr.max_cluster_budget}")
    print(f"Pressure: {kernel.resource_mgr.pressure_level:.2f}")

    kernel.ingest_input(
        "KERCA uses ergoregion reconciliation to force near-miss resolution. "
        "Tension capsules record contradictions between high-gravity capsules. "
        "Temporal reprocessing is bounded by a budget to prevent feedback explosion. "
        "Orbit 0 identity is protected by core inertia requiring multi-cycle consensus.",
        source="demo"
    )
    kernel.activate_by_query("ergoregion reconciliation tension reprocessing")

    print("\nRunning 8 cycles...")
    for _ in range(8):
        r = kernel.run_cycle()
        print(f"  Cycle {r['cycle']:>2}: {r['merges_this_cycle']} merges, "
              f"{r['tensions_created']} tensions, {r['tensions_resolved']} resolved, "
              f"{r['ergo_resolutions']} ergo, {r['reprocessed']} reproc, "
              f"{r['challenges_created']} challenges")

    final = kernel.status()
    print(f"\nFinal state:")
    print(f"  Capsules: {final['total_capsules']} ({final['shadows']} shadows)")
    print(f"  Tensions: {final['tensions']} | Challenges: {final['challenges']}")
    print(f"  Needs review: {final['needs_review']} | Reprocess exhausted: {final['reprocess_exhausted']}")
    print(f"  Orbit 0: {final['orbit0_count']}")
    print(f"  Merges: {final['routing_stats']['merges_performed']}")
    print(f"  Ergo resolutions: {final['routing_stats']['ergo_resolutions']}")
    print(f"  Tensions created: {final['routing_stats']['tensions_created']}")
    print(f"  Tensions compressed: {final['routing_stats']['tensions_compressed']}")
    print(f"  Challenges created: {final['routing_stats']['challenges_created']}")
    print(f"\nKnowledge base: {DEFAULT_KNOWLEDGE_BASE_PATH}")


if __name__ == "__main__":
    main()