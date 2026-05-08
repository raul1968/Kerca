#!/usr/bin/env python3
"""
KERCA Oracle v1.0 — GUI Launch File with Cluster Visualization
===============================================================
Complete GUI application integrating:
  • Chat interface with KERCA kernel (cluster activation, ergoregion)
  • Real-time orbital + cluster visualization
  • Cascade arc rendering between clusters
  • Capsule inspector with KERCA-specific fields (tensions, challenges, inertia)
  • Cluster panel with activation state
  • Timeline replay viewer
  • Autonomous cycle controls

KERCA-specific features:
  • Ergo lane visualization
  • Tension capsule highlighting
  • Challenge capsule tracking
  • Core inertia display
  • Reprocess budget indicators

Run: python kerca_oracle_gui.py
"""

import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Ensure kerca_kernel is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Kerca_kernel import (
    bootstrap_kernel,
    Capsule,
    CapsuleKind,
    ActivationState,
    ProcessingLane,
    ORBIT_LABELS,
    LANE_BOUNDARIES,
    Cluster,
    ClusterManager,
    KERCAKernel,
    DEFAULT_KNOWLEDGE_BASE_PATH,
    DEFAULT_TEMPORAL_STORE_PATH,
    JSON_DIR,
)

# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------

def ensure_directories() -> Dict[str, Path]:
    dirs = {
        "project_root": SCRIPT_DIR,
        "json_dir": SCRIPT_DIR / "Json",
        "ingest_dir": SCRIPT_DIR / "Json" / "ingest",
        "backup_dir": SCRIPT_DIR / "Json" / "backups",
        "data_dir": SCRIPT_DIR / "data",
        "logs_dir": SCRIPT_DIR / "logs",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QPushButton, QLabel, QSplitter, QListWidget,
        QListWidgetItem, QTabWidget, QFrame, QComboBox, QGridLayout,
        QStatusBar, QToolBar,
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QFont, QTextCursor, QColor, QPainter, QBrush, QPen,
        QRadialGradient, QPainterPath, QAction,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not found. Install with: pip install PyQt6")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Orbital + Cluster Visualizer
# ---------------------------------------------------------------------------

class OrbitalVisualizer(QWidget):
    """Real-time orbital capsule visualization with KERCA cluster rendering."""

    LANE_COLORS = {
        0: QColor(255, 215, 0, 200),
        1: QColor(100, 200, 255, 200),
        2: QColor(150, 255, 150, 200),
        3: QColor(255, 150, 150, 200),
    }

    ACTIVATION_COLORS = {
        "active": QColor(255, 255, 100, 255),
        "cooling": QColor(255, 150, 50, 180),
        "ready": QColor(100, 200, 255, 120),
        "cold": QColor(80, 80, 80, 80),
    }

    CLUSTER_BORDER_COLORS = [
        QColor(255, 200, 50, 80),
        QColor(100, 255, 200, 80),
        QColor(200, 150, 255, 80),
        QColor(255, 150, 150, 80),
        QColor(150, 255, 150, 80),
        QColor(150, 150, 255, 80),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capsules: Dict[str, Capsule] = {}
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_edges: Dict[Tuple[str, str], float] = {}
        self.selected_capsule_id: Optional[str] = None
        self.highlight_ids: List[str] = []
        self.animation_phase: float = 0.0
        self.show_labels: bool = True
        self.show_clusters: bool = True
        self.show_cascades: bool = True

        self.setMouseTracking(True)
        self._hovered_capsule_id: Optional[str] = None
        self._capsule_positions: Dict[str, Tuple[float, float]] = {}
        self._cluster_centroids: Dict[str, Tuple[float, float]] = {}

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(50)

        self.setMinimumSize(500, 450)
        self.setToolTip("KERCA Orbital Network — hover for capsule details")

    def update_data(self, capsules, clusters, cluster_edges):
        self.capsules = capsules
        self.clusters = clusters
        self.cluster_edges = cluster_edges
        self.update()

    def highlight_capsules(self, ids):
        self.highlight_ids = ids
        self.update()

    def select_capsule(self, capsule_id):
        self.selected_capsule_id = capsule_id
        self.update()

    def _animate(self):
        self.animation_phase += 0.008
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(12, 12, 22))

        cx = self.width() / 2
        cy = self.height() / 2
        max_r = min(cx, cy) * 0.82

        self._draw_lanes(painter, cx, cy, max_r)
        if self.show_clusters:
            self._draw_cluster_boundaries(painter)
        if self.show_cascades:
            self._draw_cascade_arcs(painter)
        self._draw_nucleus(painter, cx, cy)

        self._capsule_positions.clear()
        self._cluster_centroids.clear()
        if self.capsules:
            self._draw_capsules(painter, cx, cy, max_r)

        self._draw_legend(painter)
        painter.end()

    def _draw_lanes(self, painter, cx, cy, max_r):
        for level in range(4):
            bounds = LANE_BOUNDARIES.get(level, (0.5, 0.75))
            mid_r = max_r * (bounds[0] + bounds[1]) / 2
            color = QColor(self.LANE_COLORS[level])
            color.setAlpha(25)
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(cx - mid_r), int(cy - mid_r), int(mid_r * 2), int(mid_r * 2))
            label_color = QColor(self.LANE_COLORS[level])
            label_color.setAlpha(100)
            painter.setPen(label_color)
            font = QFont("Monospace", 7)
            painter.setFont(font)
            painter.drawText(int(cx + mid_r + 5), int(cy - 3), ORBIT_LABELS.get(level, f"L{level}"))

    def _draw_cluster_boundaries(self, painter):
        for i, (cluster_id, cluster) in enumerate(self.clusters.items()):
            if cluster.activation_level < 0.15:
                continue
            positions = []
            for cid in cluster.capsule_ids:
                if cid in self._capsule_positions:
                    positions.append(self._capsule_positions[cid])
            if len(positions) < 2:
                continue
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            self._cluster_centroids[cluster_id] = (avg_x, avg_y)
            max_dist = max(math.sqrt((p[0]-avg_x)**2 + (p[1]-avg_y)**2) for p in positions)
            cluster_radius = max(20, max_dist + 15)
            color_idx = i % len(self.CLUSTER_BORDER_COLORS)
            color = QColor(self.CLUSTER_BORDER_COLORS[color_idx])
            alpha = int(40 + cluster.activation_level * 80)
            color.setAlpha(min(200, alpha))
            pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            gradient = QRadialGradient(avg_x, avg_y, cluster_radius)
            glow = QColor(color)
            glow.setAlpha(int(15 + cluster.activation_level * 40))
            gradient.setColorAt(0, glow)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(int(avg_x-cluster_radius), int(avg_y-cluster_radius),
                              int(cluster_radius*2), int(cluster_radius*2))
            if cluster.activation_level > 0.3:
                painter.setPen(QColor(255, 255, 255, 150))
                painter.setFont(QFont("Monospace", 7))
                painter.drawText(int(avg_x-30), int(avg_y-cluster_radius-5), cluster.name[:20])

    def _draw_cascade_arcs(self, painter):
        for (src_id, dst_id), strength in self.cluster_edges.items():
            if src_id not in self.clusters or dst_id not in self.clusters:
                continue
            src = self.clusters[src_id]
            dst = self.clusters[dst_id]
            if src.activation_level < 0.15 and dst.activation_level < 0.15:
                continue
            if src_id not in self._cluster_centroids or dst_id not in self._cluster_centroids:
                continue
            sp = self._cluster_centroids[src_id]
            dp = self._cluster_centroids[dst_id]
            max_act = max(src.activation_level, dst.activation_level)
            alpha = int(40 + strength * 100 + max_act * 80)
            arc_color = QColor(255, 255, 150, min(255, alpha)) if max_act > 0.5 else QColor(150, 200, 255, min(180, alpha))
            pen = QPen(arc_color, 0.8 + strength * 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mx, my = (sp[0]+dp[0])/2, (sp[1]+dp[1])/2
            dx, dy = dp[0]-sp[0], dp[1]-sp[1]
            dist = math.sqrt(dx*dx+dy*dy)
            ox = -dy/dist*dist*0.3 if dist > 0 else 0
            oy = dx/dist*dist*0.3 if dist > 0 else 0
            path = QPainterPath()
            path.moveTo(sp[0], sp[1])
            path.quadTo(mx+ox, my+oy, dp[0], dp[1])
            painter.drawPath(path)

    def _draw_nucleus(self, painter, cx, cy):
        gradient = QRadialGradient(cx, cy, 25)
        gradient.setColorAt(0, QColor(255, 255, 200, 220))
        gradient.setColorAt(0.3, QColor(255, 200, 50, 160))
        gradient.setColorAt(0.7, QColor(255, 150, 0, 50))
        gradient.setColorAt(1.0, QColor(255, 100, 0, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx-18), int(cy-18), 36, 36)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setFont(QFont("Monospace", 8, QFont.Weight.Bold))
        painter.drawText(int(cx-22), int(cy+4), "KERCA")

    def _draw_capsules(self, painter, cx, cy, max_r):
        for capsule_id, capsule in self.capsules.items():
            if capsule.is_shadow or capsule.merged_into:
                continue
            bounds = LANE_BOUNDARIES.get(capsule.orbit_level, (0.5, 0.75))
            orbit_r = max_r * (bounds[0] + bounds[1]) / 2
            orbit_r -= (1.0 - capsule.gravity_score) * max_r * 0.04
            angle = capsule.orbit_angle + self.animation_phase * (0.2 + capsule.gravity_score * 0.6)
            x = cx + orbit_r * math.cos(angle)
            y = cy + orbit_r * math.sin(angle)
            self._capsule_positions[capsule_id] = (x, y)

            base_size = 5
            size = base_size + capsule.gravity_score * 10
            if capsule.activation_state == ActivationState.ACTIVE:
                size += 3

            lane_color = self.LANE_COLORS.get(capsule.orbit_level, QColor(150, 150, 150))
            if capsule.activation_state == ActivationState.ACTIVE:
                color = QColor(255, 255, 100, 255)
            elif capsule.activation_state == ActivationState.COOLING:
                color = QColor(255, 150, 50, 200)
            elif capsule.activation_state == ActivationState.READY:
                color = lane_color
                color.setAlpha(150)
            else:
                color = QColor(80, 80, 80, 100)

            # KERCA: highlight tensions and challenges
            if capsule.kind == CapsuleKind.TENSION:
                color = QColor(255, 100, 100, 220)
                size += 2
            elif capsule.kind == CapsuleKind.CHALLENGE:
                color = QColor(255, 180, 50, 220)
                size += 2
            elif capsule.needs_review:
                color = QColor(200, 200, 50, 200)

            is_sel = capsule_id == self.selected_capsule_id
            is_hl = capsule_id in self.highlight_ids
            is_hov = capsule_id == self._hovered_capsule_id

            if is_sel:
                color = QColor(255, 255, 255, 255)
                size += 5
            elif is_hl:
                color = QColor(255, 255, 120, 255)
                size += 3
            elif is_hov:
                color = color.lighter(160)
                size += 3

            if capsule.activation_state == ActivationState.ACTIVE:
                glow = QRadialGradient(x, y, size*2)
                glow.setColorAt(0, QColor(255, 255, 150, 80))
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x-size*2), int(y-size*2), int(size*4), int(size*4))

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2) if (is_sel or is_hov) else QPen(color.darker(140), 1))
            painter.drawEllipse(int(x-size/2), int(y-size/2), int(size), int(size))

            if self.show_labels and (is_hov or is_sel or capsule.gravity_score > 0.6 or capsule.kind in (CapsuleKind.TENSION, CapsuleKind.CHALLENGE)):
                painter.setPen(QColor(255, 255, 255, 160))
                painter.setFont(QFont("Monospace", 6))
                short = capsule.name[:12] + ".." if len(capsule.name) > 12 else capsule.name
                painter.drawText(int(x+size/2+3), int(y+3), short)

    def _draw_legend(self, painter):
        painter.setPen(QColor(200, 200, 200, 120))
        painter.setFont(QFont("Monospace", 7))
        ys = self.height() - 120
        xs = 10

        for level in range(4):
            color = self.LANE_COLORS[level]
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(150), 1))
            painter.drawEllipse(xs, ys+level*16, 8, 8)
            painter.setPen(QColor(200, 200, 200, 120))
            painter.drawText(xs+13, ys+level*16+8, f"Orbit {level}: {ORBIT_LABELS.get(level,'?')}")

        ys2 = ys + 70
        for i, (label, color) in enumerate([
            ("active", QColor(255,255,100)), ("cooling", QColor(255,150,50)),
            ("ready", QColor(100,200,255)), ("cold", QColor(80,80,80)),
            ("tension", QColor(255,100,100)), ("challenge", QColor(255,180,50)),
        ]):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(150), 1))
            painter.drawEllipse(xs, ys2+i*14, 7, 7)
            painter.setPen(QColor(200, 200, 200, 120))
            painter.drawText(xs+11, ys2+i*14+7, label)

        xr = self.width() - 200
        active = sum(1 for c in self.capsules.values() if c.is_active())
        tensions = sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.TENSION)
        challenges = sum(1 for c in self.capsules.values() if c.kind == CapsuleKind.CHALLENGE)
        active_cl = sum(1 for c in self.clusters.values() if c.activation_level > 0.15)
        for i, stat in enumerate([
            f"Caps: {active}a | T:{tensions} | C:{challenges}",
            f"Clusters: {active_cl} active / {len(self.clusters)} total",
        ]):
            painter.drawText(xr, ys+i*16+8, stat)

    def mouseMoveEvent(self, event):
        mx, my = event.position().x(), event.position().y()
        self._hovered_capsule_id = None
        for cid, (px, py) in self._capsule_positions.items():
            if math.sqrt((mx-px)**2 + (my-py)**2) < 15:
                self._hovered_capsule_id = cid
                break
        if self._hovered_capsule_id:
            cap = self.capsules.get(self._hovered_capsule_id)
            if cap:
                self.setToolTip(
                    f"{cap.name}\nKind: {cap.kind.value}\n"
                    f"Orbit: {cap.orbit_level} ({ORBIT_LABELS.get(cap.orbit_level,'?')})\n"
                    f"Activation: {cap.activation_state.value}\n"
                    f"Gravity: {cap.gravity_score:.3f}\n"
                    f"Lane: {cap.processing_lane.value}\n"
                    f"Tensions: {len(cap.tension_pairs)}\n"
                    f"Reprocess: {cap.reprocess_count}/{cap.max_reprocess}\n"
                    f"Core inertia: {cap.core_inertia}" + (f"\n⚠ NEEDS REVIEW" if cap.needs_review else "")
                )
        self.update()

    def mousePressEvent(self, event):
        if self._hovered_capsule_id:
            self.selected_capsule_id = self._hovered_capsule_id
            self.update()


# ---------------------------------------------------------------------------
# Chat Widget
# ---------------------------------------------------------------------------

class ChatWidget(QWidget):
    message_sent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.message_history: List[Dict[str, str]] = []

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit { background-color: #1a1a2e; color: #e0e0e0; border: 1px solid #333;
                       border-radius: 4px; padding: 8px; font-family: 'Monospace'; font-size: 12px; }
        """)
        layout.addWidget(self.chat_display, 1)

        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background-color: #16213e; border: 1px solid #333; border-radius: 4px; }")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(5, 5, 5, 5)

        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(70)
        self.input_field.setPlaceholderText("Query KERCA: 'status', 'dynamic routing', 'tension capsules'...")
        self.input_field.setStyleSheet("""
            QTextEdit { background-color: #0f0f23; color: #e0e0e0; border: none; font-family: 'Monospace'; font-size: 12px; }
        """)
        input_layout.addWidget(self.input_field)

        btn_layout = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_message)
        self.send_button.setStyleSheet("""
            QPushButton { background-color: #0f3460; color: #e0e0e0; border: 1px solid #1a5276;
                         border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #1a5276; }
        """)
        btn_layout.addWidget(self.send_button)
        btn_layout.addStretch()
        input_layout.addLayout(btn_layout)
        layout.addWidget(input_frame)
        self.input_field.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text: return
        self.add_message("You", text, "#4fc3f7")
        self.input_field.clear()
        self.message_sent.emit(text)

    def add_message(self, sender, message, color="#e0e0e0"):
        self.message_history.append({"sender": sender, "content": message})
        msg = message.replace("\n", "<br>")
        formatted = f'<p><b style="color:{color}">{sender}:</b><br>{msg}</p><hr style="border-color:#333">'
        self.chat_display.append(formatted)
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)


# ---------------------------------------------------------------------------
# Capsule Inspector
# ---------------------------------------------------------------------------

class CapsuleInspector(QWidget):
    capsule_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.capsules: Dict[str, Capsule] = {}

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Active", "Ready", "Cooling", "Cold",
                                    "Core (Orbit 0)", "Tensions", "Challenges", "Needs Review"])
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        self.filter_combo.setStyleSheet("""
            QComboBox { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #333; border-radius: 4px; padding: 4px; }
        """)
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)

        self.capsule_list = QListWidget()
        self.capsule_list.setStyleSheet("""
            QListWidget { background-color: #1a1a2e; color: #e0e0e0; border: 1px solid #333; border-radius: 4px; }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #2a2a4a; }
            QListWidget::item:selected { background-color: #0f3460; }
            QListWidget::item:hover { background-color: #16213e; }
        """)
        self.capsule_list.itemClicked.connect(self._on_click)
        layout.addWidget(self.capsule_list)

        self.detail_display = QTextEdit()
        self.detail_display.setReadOnly(True)
        self.detail_display.setStyleSheet("""
            QTextEdit { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #333;
                       border-radius: 4px; padding: 8px; font-family: 'Monospace'; font-size: 11px; }
        """)
        layout.addWidget(self.detail_display)

    def update_capsules(self, capsules):
        self.capsules = capsules
        self._apply_filter(self.filter_combo.currentText())

    def _apply_filter(self, filter_text):
        self.capsule_list.clear()
        sorted_caps = sorted(
            [c for c in self.capsules.values() if not c.is_shadow and not c.merged_into],
            key=lambda c: (c.orbit_level, -c.gravity_score)
        )
        act_icons = {ActivationState.ACTIVE: "⚡", ActivationState.COOLING: "🌙",
                     ActivationState.READY: "●", ActivationState.COLD: "○"}

        for capsule in sorted_caps:
            if filter_text == "Active" and capsule.activation_state != ActivationState.ACTIVE: continue
            if filter_text == "Ready" and capsule.activation_state != ActivationState.READY: continue
            if filter_text == "Cooling" and capsule.activation_state != ActivationState.COOLING: continue
            if filter_text == "Cold" and capsule.activation_state != ActivationState.COLD: continue
            if filter_text == "Core (Orbit 0)" and capsule.orbit_level != 0: continue
            if filter_text == "Tensions" and capsule.kind != CapsuleKind.TENSION: continue
            if filter_text == "Challenges" and capsule.kind != CapsuleKind.CHALLENGE: continue
            if filter_text == "Needs Review" and not capsule.needs_review: continue

            icon = act_icons.get(capsule.activation_state, "?")
            flags = ""
            if capsule.kind == CapsuleKind.TENSION: flags += " ⚠"
            if capsule.kind == CapsuleKind.CHALLENGE: flags += " 🔔"
            if capsule.needs_review: flags += " 📋"
            if capsule.reprocess_exhausted: flags += " ⛔"

            item = QListWidgetItem(f"{icon} [{capsule.orbit_level}] {capsule.name[:40]} (g:{capsule.gravity_score:.2f}){flags}")
            item.setData(Qt.ItemDataRole.UserRole, capsule.id)
            colors = {0: QColor(255,215,0), 1: QColor(100,200,255), 2: QColor(150,255,150), 3: QColor(255,150,150)}
            if capsule.kind == CapsuleKind.TENSION: color = QColor(255,100,100)
            elif capsule.kind == CapsuleKind.CHALLENGE: color = QColor(255,180,50)
            else: color = colors.get(capsule.orbit_level, QColor(200,200,200))
            item.setForeground(color)
            self.capsule_list.addItem(item)

    def _on_click(self, item):
        capsule_id = item.data(Qt.ItemDataRole.UserRole)
        capsule = self.capsules.get(capsule_id)
        if not capsule: return
        self.capsule_selected.emit(capsule_id)

        details = f"""
<b style="color:gold">Name:</b> {capsule.name}
<b style="color:gold">Kind:</b> {capsule.kind.value}
<b style="color:gold">Orbit:</b> {capsule.orbit_level} ({ORBIT_LABELS.get(capsule.orbit_level,'?')})
<b style="color:gold">Activation:</b> {capsule.activation_state.value}
<b style="color:gold">Lane:</b> {capsule.processing_lane.value}
<b style="color:gold">Gravity:</b> {capsule.gravity_score:.4f}
<b style="color:gold">Agreement:</b> {capsule.agreement_score:.4f}
<b style="color:gold">Confidence:</b> {capsule.confidence:.4f}
<b style="color:gold">Usage:</b> {capsule.usage_count}
<b style="color:gold">Sustain:</b> {capsule.sustain_cycles}
<b style="color:gold">Cluster:</b> {capsule.cluster_id or 'None'}
<b style="color:gold">Shadows:</b> {len(capsule.shadows)}

<b style="color:#ff6b6b">KERCA Fields:</b>
<b style="color:gold">Tension pairs:</b> {len(capsule.tension_pairs)}
<b style="color:gold">Needs review:</b> {capsule.needs_review}
<b style="color:gold">Reprocess:</b> {capsule.reprocess_count}/{capsule.max_reprocess} {'⛔ EXHAUSTED' if capsule.reprocess_exhausted else ''}
<b style="color:gold">Core inertia:</b> {capsule.core_inertia} (req: {capsule.core_consensus_required})
<b style="color:gold">Core mods:</b> {capsule.core_modification_count}
<b style="color:gold">Ergo processed:</b> {capsule.ergo_processed}

<b style="color:#4fc3f7">Content:</b>
{', '.join(capsule.content.get('keywords', [])[:20])}

<b style="color:#4fc3f7">Description:</b>
{capsule.content.get('description', 'No description')[:300]}
"""
        self.detail_display.setHtml(details)


# ---------------------------------------------------------------------------
# Main Oracle GUI
# ---------------------------------------------------------------------------

class OracleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dirs = ensure_directories()
        self.kernel = bootstrap_kernel(load_existing=True)

        self.auto_cycle_timer = QTimer(self)
        self.auto_cycle_timer.timeout.connect(self._auto_cycle)
        self.auto_cycle_enabled = False
        self.query_results_ids: List[str] = []

        self._init_ui()
        self._update_all_displays()
        self._show_welcome()

    def _init_ui(self):
        self.setWindowTitle("KERCA Oracle v1.0 — Kerr Engine for Routing and Capsule Agreement")
        self.setGeometry(60, 60, 1500, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a1a; }
            QWidget { color: #e0e0e0; font-family: 'Monospace', 'Courier New'; }
            QSplitter::handle { background-color: #333; width: 2px; }
            QTabWidget::pane { background-color: #1a1a2e; border: 1px solid #333; }
            QTabBar::tab { background-color: #16213e; color: #e0e0e0; padding: 8px 16px; border: 1px solid #333; border-radius: 4px; }
            QTabBar::tab:selected { background-color: #0f3460; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._create_toolbar()

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_widget = ChatWidget()
        self.chat_widget.message_sent.connect(self._handle_chat_message)
        left_layout.addWidget(self.chat_widget, 2)

        control_tabs = QTabWidget()
        control_tabs.setMaximumHeight(280)

        self.capsule_inspector = CapsuleInspector()
        self.capsule_inspector.capsule_selected.connect(lambda cid: self.orbital_visualizer.select_capsule(cid))
        control_tabs.addTab(self.capsule_inspector, "Capsules")

        self.query_results_display = QTextEdit()
        self.query_results_display.setReadOnly(True)
        self.query_results_display.setStyleSheet("""
            QTextEdit { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #333; font-family: 'Monospace'; font-size: 11px; }
        """)
        control_tabs.addTab(self.query_results_display, "Query Results")

        self.timeline_display = QTextEdit()
        self.timeline_display.setReadOnly(True)
        self.timeline_display.setStyleSheet("""
            QTextEdit { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #333; font-family: 'Monospace'; font-size: 11px; }
        """)
        control_tabs.addTab(self.timeline_display, "Timeline")

        left_layout.addWidget(control_tabs)
        main_splitter.addWidget(left_panel)

        # RIGHT
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.orbital_visualizer = OrbitalVisualizer()
        right_layout.addWidget(self.orbital_visualizer, 3)

        status_frame = QFrame()
        status_frame.setStyleSheet("QFrame { background-color: #16213e; border: 1px solid #333; border-radius: 4px; padding: 8px; }")
        status_layout = QGridLayout(status_frame)
        status_layout.setContentsMargins(8, 8, 8, 8)

        self.status_labels = {}
        items = [
            (0,0,"Total:","total"),(0,1,"Active:","active"),(0,2,"Ready:","ready"),
            (0,3,"Shadows:","shadows"),(0,4,"Orbit0:","orbit0"),
            (1,0,"Clusters:","clusters"),(1,1,"Active Cl:","active_cl"),
            (1,2,"Tensions:","tensions"),(1,3,"Challenges:","challenges"),
            (1,4,"Review:","review"),
            (2,0,"Cycles:","cycles"),(2,1,"Merges:","merges"),
            (2,2,"Ergo:","ergo"),(2,3,"Reproc:","reproc"),
            (2,4,"Pressure:","pressure"),
        ]
        for row, col, label, key in items:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888; font-size: 9px;")
            val = QLabel("--")
            val.setStyleSheet("color: #4fc3f7; font-size: 9px; font-weight: bold;")
            self.status_labels[key] = val
            status_layout.addWidget(lbl, row, col*2)
            status_layout.addWidget(val, row, col*2+1)

        right_layout.addWidget(status_frame)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([700, 800])
        main_layout.addWidget(main_splitter)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar { background-color: #0a0a1a; color: #888; border-top: 1px solid #333; }")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("KERCA Oracle ready. Type 'help' for commands.")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._update_all_displays)
        self.refresh_timer.start(3000)

    def _create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar { background-color: #16213e; border: 1px solid #333; border-radius: 4px; padding: 4px; }
            QToolButton { background-color: #0f3460; color: #e0e0e0; border: 1px solid #1a5276; border-radius: 4px; padding: 6px 12px; }
            QToolButton:hover { background-color: #1a5276; }
        """)
        self.addToolBar(toolbar)

        toolbar.addAction("Run Cycle").triggered.connect(lambda: self._handle_chat_message("run cycle"))
        self.auto_cycle_btn = toolbar.addAction("Auto Cycle")
        self.auto_cycle_btn.setCheckable(True)
        self.auto_cycle_btn.triggered.connect(self._toggle_auto_cycle)
        toolbar.addSeparator()
        toolbar.addAction("Status").triggered.connect(lambda: self._handle_chat_message("status"))
        toolbar.addAction("Timeline").triggered.connect(lambda: self._handle_chat_message("timeline"))
        toolbar.addSeparator()
        toolbar.addAction("Save").triggered.connect(lambda: self._handle_chat_message("save"))
        toolbar.addAction("Help").triggered.connect(lambda: self._handle_chat_message("help"))

    def _handle_chat_message(self, message):
        lower = message.lower().strip()
        if lower in ("help", "?"): response = self._get_help()
        elif lower in ("status", "system status"): response = self._get_status()
        elif lower.startswith("query ") or lower.startswith("search "):
            term = message.split(" ", 1)[1] if " " in message else message
            response = self._handle_query(term)
        elif lower in ("run cycle", "cycle"): response = self._run_cycle()
        elif lower in ("save", "persist"):
            self.kernel.store.save(self.kernel.capsules, clusters=self.kernel.cluster_mgr.to_dict())
            response = "State saved."
        elif lower in ("timeline", "frames"): response = self._show_timeline()
        else: response = self._handle_query(message)

        self.chat_widget.add_message("Oracle", response, "#80cbc4")
        self._update_all_displays()

    def _get_help(self):
        return """
<b style="color:gold">KERCA Oracle v1.0 — Commands</b>

<b style="color:#4fc3f7">Queries (trigger cluster activation):</b>
  • <b>status</b> — Full system status
  • <b>dynamic routing vs EM routing</b>
  • <b>capsule efficiency</b>
  • <b>routing iterations</b>
  • <b>matrix capsules</b>
  • <b>tension capsules</b>
  • <b>orbit 0 capsules</b>
  • <b>challenges to core</b>
  • <b>run cycle</b> — Execute one cognition cycle
  • <b>save</b> — Persist to disk
  • <b>timeline</b> — View recent frames

<b style="color:#4fc3f7">KERCA-specific queries:</b>
  • Tension capsules shown in <b style="color:#ff6b6b">red</b>
  • Challenge capsules shown in <b style="color:#ffb432">orange</b>
  • Needs-review capsules shown in <b style="color:#c8c832">yellow</b>
  • Orbit 0 capsules have core inertia protection
"""

    def _get_status(self):
        try:
            s = self.kernel.status()
            return f"""
<b style="color:gold">═══ KERCA System Status ═══</b>

<b>Capsules:</b> Total: {s['total_capsules']} | Active: {s['active_capsules']} | Ready: {s['ready_capsules']}
  Cooling: {s['cooling_capsules']} | Cold: {s['cold_capsules']} | Shadows: {s['shadows']}

<b>KERCA:</b> Tensions: {s['tensions']} | Challenges: {s['challenges']} | Needs Review: {s['needs_review']}
  Reprocess Exhausted: {s['reprocess_exhausted']} | Orbit 0: {s['orbit0_count']}

<b>Orbits:</b> L0: {s['orbit_distribution'].get(0,0)} | L1: {s['orbit_distribution'].get(1,0)} | L2: {s['orbit_distribution'].get(2,0)} | L3: {s['orbit_distribution'].get(3,0)}

<b>Clusters:</b> {s['total_clusters']} total | {s['active_clusters']} active | {s['ready_clusters']} ready

<b>Routing:</b> Cycles: {s['cycle_count']} | Merges: {s['routing_stats']['merges_performed']}
  Ergo: {s['routing_stats']['ergo_resolutions']} | Tensions: {s['routing_stats']['tensions_created']}
  Challenges: {s['routing_stats']['challenges_created']}

<b>Resources:</b> Budget: {s['cluster_budget']}/{s['resource_stats']['max_cluster_budget']} | Pressure: {s['pressure']:.2f}
"""
        except Exception as e:
            return f"Error: {e}"

    def _handle_query(self, query_text):
        try:
            for prefix in ("query ", "search ", "find "):
                if query_text.lower().startswith(prefix):
                    query_text = query_text[len(prefix):]
                    break
            results = self.kernel.query(query_text, top_k=8, activate=True)
            self.query_results_ids = [r["id"] for r in results]
            self.orbital_visualizer.highlight_capsules(self.query_results_ids)

            if not results:
                return f"No results for '{query_text}'."

            lines = [f"<b style='color:gold'>Query: '{query_text}' — {len(results)} results</b>\n"]
            for i, r in enumerate(results, 1):
                flags = ""
                if r.get('tension'): flags += " ⚠"
                if r.get('needs_review'): flags += " 📋"
                if r.get('reprocess_exhausted'): flags += " ⛔"
                act_icon = {"active":"⚡","cooling":"🌙","ready":"●","cold":"○"}.get(r['activation'],'?')
                lines.append(
                    f"{i}. {act_icon} <b>[{r['orbit_label']}]</b> {r['name']} "
                    f"(g:{r['gravity']:.3f}, s:{r['score']:.3f}){flags}"
                )
                lines.append(f"   <span style='color:#888'>{r['content_summary']}</span>")

            self.kernel.ingest_input(query_text, source="user_query")
            r = self.kernel.run_cycle()
            lines.append(f"\n<span style='color:#888'>Clusters activated • Cycle {r['cycle']} • "
                        f"{r['active_clusters']} clusters • {r['merges_this_cycle']} merges</span>")
            return "\n".join(lines)
        except Exception as e:
            return f"Query error: {e}"

    def _run_cycle(self):
        try:
            r = self.kernel.run_cycle()
            self._update_all_displays()
            return f"""
<b style="color:#4fc3f7">Cycle {r['cycle']}</b>
  Active: {r['active_capsules']} caps | {r['active_clusters']} clusters
  Merges: {r['merges_this_cycle']} | Shadows: {r['shadows_created']}
  Tensions: {r['tensions_created']} | Ergo: {r['ergo_resolutions']}
  Reprocessed: {r['reprocessed']} | Challenges: {r['challenges_created']}
  Max gravity: {r['max_gravity']:.3f} | Budget: {r['cluster_budget']}
"""
        except Exception as e:
            return f"Cycle error: {e}"

    def _show_timeline(self):
        try:
            frames = self.kernel.timeline.replay(-10)
            lines = [f"<b style='color:gold'>═══ Timeline (last {len(frames)}) ═══</b>\n"]
            for f in frames:
                lines.append(
                    f"[{f['timestamp'][:19]}] Caps:{f['active_count']} | "
                    f"Cl:{f.get('cluster_count','?')} | Merges:{f['merges_this_cycle']} | "
                    f"Tens:{f.get('tensions_created',0)} | Ergo:{f.get('ergo_resolutions',0)} | "
                    f"Reproc:{f.get('reprocessed_count',0)}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Timeline error: {e}"

    def _toggle_auto_cycle(self):
        self.auto_cycle_enabled = self.auto_cycle_btn.isChecked()
        if self.auto_cycle_enabled:
            self.auto_cycle_timer.start(10000)
            self.status_bar.showMessage("Auto-cycle enabled (10s)")
        else:
            self.auto_cycle_timer.stop()
            self.status_bar.showMessage("Auto-cycle disabled")

    def _auto_cycle(self):
        try:
            self.kernel.run_cycle()
            self._update_all_displays()
            s = self.kernel.status()
            self.status_bar.showMessage(f"Auto-cycle {s['cycle_count']} | {s['active_clusters']} clusters | T:{s['tensions']} C:{s['challenges']}")
        except: pass

    def _update_all_displays(self):
        self.orbital_visualizer.update_data(
            self.kernel.capsules,
            self.kernel.cluster_mgr.clusters,
            self.kernel.cluster_mgr.cluster_edges,
        )
        self.capsule_inspector.update_capsules(self.kernel.capsules)

        try:
            s = self.kernel.status()
            self.status_labels["total"].setText(str(s['total_capsules']))
            self.status_labels["active"].setText(str(s['active_capsules']))
            self.status_labels["ready"].setText(str(s['ready_capsules']))
            self.status_labels["shadows"].setText(str(s['shadows']))
            self.status_labels["orbit0"].setText(str(s['orbit0_count']))
            self.status_labels["clusters"].setText(str(s['total_clusters']))
            self.status_labels["active_cl"].setText(str(s['active_clusters']))
            self.status_labels["tensions"].setText(str(s['tensions']))
            self.status_labels["challenges"].setText(str(s['challenges']))
            self.status_labels["review"].setText(str(s['needs_review']))
            self.status_labels["cycles"].setText(str(s['cycle_count']))
            self.status_labels["merges"].setText(str(s['routing_stats']['merges_performed']))
            self.status_labels["ergo"].setText(str(s['routing_stats']['ergo_resolutions']))
            self.status_labels["reproc"].setText(str(s['timeline_stats'].get('total_reprocessed',0)))
            self.status_labels["pressure"].setText(f"{s['pressure']:.2f}")
        except: pass

    def _show_welcome(self):
        s = self.kernel.status()
        self.chat_widget.add_message("Oracle", f"""
<b style="color:gold">🪐 KERCA Oracle v1.0 — Online</b>

<b>System Ready:</b>
  Capsules: {s['total_capsules']} ({s['active_capsules']} active, {s['ready_capsules']} ready)
  Orbits: {s['orbit_distribution']}
  Clusters: {s['total_clusters']} ({s['active_clusters']} active)
  Tensions: {s['tensions']} | Challenges: {s['challenges']}
  Orbit 0: {s['orbit0_count']}

<b>Type a query</b> to trigger chain-lightning cluster activation.
Tension capsules shown in <b style="color:#ff6b6b">red</b>.
Challenge capsules shown in <b style="color:#ffb432">orange</b>.

Type <b>'help'</b> for commands.
""", "#80cbc4")

    def closeEvent(self, event):
        self.kernel.store.save(self.kernel.capsules, clusters=self.kernel.cluster_mgr.to_dict())
        self.auto_cycle_timer.stop()
        self.refresh_timer.stop()
        event.accept()


def main():
    print("=" * 60)
    print("KERCA Oracle GUI v1.0")
    print("=" * 60)
    ensure_directories()
    print(f"Knowledge base: {DEFAULT_KNOWLEDGE_BASE_PATH}")
    print("Starting GUI...")
    app = QApplication(sys.argv)
    app.setApplicationName("KERCA Oracle")
    window = OracleGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()