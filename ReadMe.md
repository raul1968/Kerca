# KERCA — Kerr Engine for Routing and Capsule Agreement

**Version:** 1.0  
**Status:** Active — 251 papers ingested, 38,465 capsules, 65 clusters, 510 Orbit 0

---

## What Is KERCA?

KERCA is a deterministic, capsule-based cognitive engine that organizes knowledge through routing-by-agreement, orbital memory layers, cluster activation, and rotational feedback via an ergoregion reconciliation zone.

It ingests research papers, builds a semantic graph, and lets you query the emergent structure.

---

## Quick Start

```powershell
# Install dependencies
pip install numpy PyPDF2 PyQt6 psutil

# Full pipeline (fresh start)
python ingest_pdfs.py --dir "papers"
python initial_cognition.py
python promote_merged.py
python fix_embeddings.py
python activate_clusters.py
python kerca_oracle_gui.py

File Reference
File	Purpose	Run When
kerca_kernel.py	Core engine — Capsule, Cluster, Routing, Gravity, Ergoregion, Timeline	Never run directly (imported)
ingest_pdfs.py	Extract text from PDFs, chunk, create ingestion capsules	STEP 1 — Fresh start or adding new papers
initial_cognition.py	Warm capsules, lower threshold, build agreement graph, first merges	STEP 2 — After ingestion
promote_merged.py	Move merged capsules from Orbit 3 → Orbit 2	STEP 3 — After initial cognition
fix_embeddings.py	Recompute broken embeddings, force keyword clusters, additional merges	STEP 4 — After promotion
activate_clusters.py	Fire all clusters, run KERCA cycles with ergoregion + reprocessing	STEP 5 — Final processing
kerca_oracle_gui.py	GUI: chat queries, orbital visualizer, capsule inspector, timeline	STEP 6 — Exploration
Pipeline Walkthrough
Step 1: Ingest Papers
powershell
python ingest_pdfs.py --dir "papers"
Extracts text from all PDFs in the papers/ directory, splits into 1,500-character chunks, creates capsules at Orbit 3. Runs periodic cycles during ingestion. Saves to Json/kerca_knowledge_base.json.

Step 2: Initial Cognition Push
powershell
python initial_cognition.py
Warms all capsules to READY state, temporarily lowers routing threshold to 0.30, processes in batches of 500 to build the agreement graph, runs refinement cycles at normal threshold.

Step 3: Promote Merged Capsules
powershell
python promote_merged.py
Moves merged capsules from Orbit 3 to Orbit 2 with boosted gravity so they participate in cluster formation and higher-orbit routing.

Step 4: Fix Embeddings + Force Clusters
powershell
python fix_embeddings.py
Recomputes embeddings for merged capsules (the weighted average from merge() degrades with zero usage counts), re-runs routing, and forces keyword-based clusters using shared substantive keywords as a fallback when the agreement graph doesn't produce clusters.

Step 5: Activate Clusters
powershell
python activate_clusters.py
Fires all clusters with cascade activation, runs 8 cognition cycles including ergoregion reconciliation (every 5th cycle), temporal reprocessing (every 12th cycle), and Orbit 0 challenge evaluation (every 20th cycle).

Step 6: Explore
powershell
python kerca_oracle_gui.py
Launches the GUI. Type queries to trigger chain-lightning cluster activation. Watch the orbital visualizer for cluster boundaries, cascade arcs, and activation glow.

GUI Reference
Chat Commands
Command	What It Does
help	Show available commands
status	Full system status (capsules, orbits, clusters, tensions, routing stats)
query <text>	Search capsules and activate relevant clusters
run cycle	Execute one cognition cycle manually
save	Persist capsules and clusters to disk
timeline	View recent temporal frames
Queries to Try
text
dynamic routing vs EM routing
capsule efficiency
routing iterations
matrix capsules
part-whole hierarchy
capsules without routing
medical imaging capsule networks
text classification capsules
sentiment analysis capsule
hinton matrix capsules EM
stacked capsule autoencoder
Visualizer Legend
Color	Meaning
Gold (Orbit 0)	Core identity
Blue (Orbit 1)	Active reasoning
Green (Orbit 2)	Working memory
Red (Orbit 3)	Raw ingestion
Bright yellow (ACTIVE)	Currently firing
Orange (COOLING)	Recently active
Red capsule	Tension capsule (contradiction)
Orange capsule	Challenge capsule (Orbit 0 challenge)
Yellow capsule	Needs review
Colored rings	Cluster boundaries
Curved lines	Cascade arcs between clusters
Architecture
text
                         ┌──────────────────────┐
                         │   ORBIT 0: CORE       │
                         │   Stable identity     │
                         │   Protected by inertia│
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │   ERGOREGION (Orbit 0.5)                  │
              │   Forced reconciliation zone               │
              │   • Near-miss resolution                   │
              │   • Tension capsule creation               │
              │   • Significance gate (gravity ≥ 0.25)     │
              │   • Tension compression + stale resolution │
              └─────────────────────┬─────────────────────┘
                                    │
                         ┌──────────┴───────────┐
                         │   ORBIT 1-2           │
                         │   Active reasoning    │
                         │   Working memory      │
                         │   Cluster activation  │
                         │   Cascade triggering  │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │   TEMPORAL REPROCESSING                   │
              │   • Re-injects past capsules              │
              │   • Bounded by budget (max 3)             │
              │   • Decay factor on re-entry              │
              │   • Orbit 0 excluded                     │
              └─────────────────────┬─────────────────────┘
                                    │
                         ┌──────────┴───────────┐
                         │   ORBIT 3             │
                         │   Raw ingestion       │
                         └──────────────────────┘
Risk Mitigations
Risk	Mechanism
Feedback explosion	Reprocessing budget (max 3 per capsule) + decay factor
Tension proliferation	Significance gate (gravity ≥ 0.25) + tension merging + stale resolution
Orbit 0 contamination	Core inertia (multi-cycle consensus) + external challenge capsules
Current State (251 Papers)
Metric	Value
Total capsules	38,465
Active capsules	Varies (0-139 depending on cycle state)
Ready capsules	~38,300
Shadows (merged)	36,160
Orbit 0	510
Orbit 1	170
Orbit 2	290
Orbit 3	1,335
Clusters	65 (keyword-based)
Tensions	0
Challenges	0
Merges total	5,000+ across all pipeline steps
Hardware Scaling
Hardware	Cluster Budget	Behavior
8GB RAM, 4 cores	~5-10 clusters	Light queries, avoid giant clusters
16GB RAM, 8 cores	~15-20 clusters	Moderate queries
64GB RAM, i5	24 clusters	Large clusters may cause brief freezes
128GB RAM, 32-core Threadripper	50-100+ clusters	Full parallel activation, real-time visualization
Design Principles
❌ No gradient descent

❌ No neural networks

❌ No backpropagation

✅ Deterministic routing-by-agreement

✅ Symbolic semantic capsules

✅ Temporal graph memory with replay

✅ Rotational feedback through ergoregion

✅ Bounded reprocessing with decay

✅ Core inertia for identity protection

✅ Hardware-aware adaptive execution

Lineage
text
ROCA v1.0 → v2.0 → v3.0 (Clusters) → KERCA v1.0 (Rotational Feedback)
License
Research project. Open for collaboration.