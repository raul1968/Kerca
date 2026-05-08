#!/usr/bin/env python3
"""
ROCA-CAE GUI - Standalone Animation System
===========================================
Complete character animation system with:
- Pose creation and editing
- Motion sequences
- Pose interpolation
- Timeline playback

This GUI works independently - no special kernel required.
"""

import sys
import math
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter, QListWidget,
    QListWidgetItem, QTabWidget, QFrame, QGridLayout,
    QStatusBar, QToolBar, QSlider, QDoubleSpinBox,
    QGroupBox, QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QBrush, QPen,
    QRadialGradient, QAction, QPainterPath
)


# ============================================================================
# Animation Data Structures
# ============================================================================

class PoseType(str, Enum):
    IDLE = "idle"
    WALK = "walk"
    RUN = "run"
    JUMP = "jump"
    ATTACK = "attack"
    DEFEND = "defend"
    SIT = "sit"
    CUSTOM = "custom"


class MotionType(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Pose:
    """A character pose with joint positions."""
    id: str
    name: str
    pose_type: PoseType
    joints: Dict[str, Vector3] = field(default_factory=dict)
    duration: float = 1.0
    
    def interpolate_to(self, other: 'Pose', t: float) -> 'Pose':
        """Interpolate between two poses."""
        t = max(0.0, min(1.0, t))
        new_joints = {}
        
        all_joints = set(self.joints.keys()) | set(other.joints.keys())
        for jid in all_joints:
            j1 = self.joints.get(jid, Vector3())
            j2 = other.joints.get(jid, Vector3())
            new_joints[jid] = Vector3(
                x=j1.x * (1-t) + j2.x * t,
                y=j1.y * (1-t) + j2.y * t,
                z=j1.z * (1-t) + j2.z * t
            )
        
        return Pose(
            id=f"interp_{self.id}_{other.id}",
            name=f"{self.name}→{other.name}",
            pose_type=self.pose_type,
            joints=new_joints,
            duration=self.duration * (1-t) + other.duration * t
        )


@dataclass
class Motion:
    """A motion sequence."""
    id: str
    name: str
    motion_type: MotionType
    poses: List[Pose]
    duration: float = 2.0
    loop: bool = False
    
    def get_pose_at_time(self, time: float) -> Optional[Pose]:
        """Get pose at specific time."""
        if not self.poses:
            return None
        
        if time <= 0:
            return self.poses[0]
        if time >= self.duration:
            return self.poses[-1] if not self.loop else self.poses[0]
        
        # Find which pose pair to interpolate
        t_per_pose = self.duration / len(self.poses)
        idx = int(time / t_per_pose)
        idx = min(idx, len(self.poses) - 2)
        
        local_t = (time - idx * t_per_pose) / t_per_pose
        
        # Apply easing
        if self.motion_type == MotionType.EASE_IN:
            local_t = local_t ** 2
        elif self.motion_type == MotionType.EASE_OUT:
            local_t = 1 - (1 - local_t) ** 2
        elif self.motion_type == MotionType.EASE_IN_OUT:
            local_t = 0.5 - 0.5 * math.cos(math.pi * local_t)
        elif self.motion_type == MotionType.BOUNCE:
            local_t = abs(math.sin(local_t * math.pi * 2))
        
        return self.poses[idx].interpolate_to(self.poses[idx + 1], local_t)


# ============================================================================
# Pose Visualizer
# ============================================================================

class PoseVisualizer(QWidget):
    """2D visualization of character poses."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pose: Optional[Pose] = None
        self.current_motion: Optional[Motion] = None
        self.animation_time = 0.0
        self.is_playing = False
        self.joint_positions: Dict[str, QPointF] = {}
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        
        self.setMinimumSize(500, 450)
        self.setStyleSheet("background-color: #1a1a2e; border: 1px solid #333; border-radius: 8px;")
    
    def set_pose(self, pose: Pose):
        """Set and display a pose."""
        self.current_pose = pose
        self.current_motion = None
        self.is_playing = False
        self.anim_timer.stop()
        self._compute_joint_positions()
        self.update()
    
    def set_motion(self, motion: Motion):
        """Set motion for playback."""
        self.current_motion = motion
        self.animation_time = 0.0
        if motion and motion.poses:
            self.current_pose = motion.poses[0]
            self._compute_joint_positions()
        self.update()
    
    def play(self):
        """Play motion."""
        if self.current_motion and self.current_motion.poses:
            self.is_playing = True
            self.anim_timer.start(16)  # ~60fps
    
    def stop(self):
        """Stop motion."""
        self.is_playing = False
        self.anim_timer.stop()
        self.animation_time = 0.0
        if self.current_motion and self.current_motion.poses:
            self.current_pose = self.current_motion.poses[0]
            self._compute_joint_positions()
            self.update()
    
    def _animate(self):
        """Animate the current motion."""
        if not self.is_playing or not self.current_motion:
            return
        
        self.animation_time += 0.033  # ~30fps increment
        if self.animation_time > self.current_motion.duration:
            if self.current_motion.loop:
                self.animation_time = 0.0
            else:
                self.stop()
                return
        
        pose = self.current_motion.get_pose_at_time(self.animation_time)
        if pose:
            self.current_pose = pose
            self._compute_joint_positions()
            self.update()
    
    def _compute_joint_positions(self):
        """Convert 3D joint positions to 2D screen positions."""
        if not self.current_pose:
            return
        
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        scale = 80
        
        # Skeleton bone connections
        self.bones = [
            ("root", "hip"), ("hip", "spine"), ("spine", "chest"),
            ("chest", "neck"), ("neck", "head"),
            ("chest", "shoulder_L"), ("shoulder_L", "elbow_L"), ("elbow_L", "hand_L"),
            ("chest", "shoulder_R"), ("shoulder_R", "elbow_R"), ("elbow_R", "hand_R"),
            ("hip", "hip_L"), ("hip_L", "knee_L"), ("knee_L", "foot_L"),
            ("hip", "hip_R"), ("hip_R", "knee_R"), ("knee_R", "foot_R"),
        ]
        
        self.joint_positions.clear()
        
        # Default positions if joints missing
        default_positions = {
            "root": (0, 0), "hip": (0, 0.2), "spine": (0, 0.4), "chest": (0, 0.6),
            "neck": (0, 0.75), "head": (0, 0.9),
            "shoulder_L": (-0.3, 0.55), "elbow_L": (-0.5, 0.5), "hand_L": (-0.6, 0.45),
            "shoulder_R": (0.3, 0.55), "elbow_R": (0.5, 0.5), "hand_R": (0.6, 0.45),
            "hip_L": (-0.25, 0.15), "knee_L": (-0.3, -0.1), "foot_L": (-0.35, -0.3),
            "hip_R": (0.25, 0.15), "knee_R": (0.3, -0.1), "foot_R": (0.35, -0.3),
        }
        
        for joint_name, (dx, dy) in default_positions.items():
            if joint_name in self.current_pose.joints:
                j = self.current_pose.joints[joint_name]
                x = cx + j.x * scale
                y = cy - j.y * scale + j.z * scale * 0.5
            else:
                x = cx + dx * scale
                y = cy - dy * scale
            self.joint_positions[joint_name] = QPointF(x, y)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(18, 18, 30))
        
        # Draw grid
        painter.setPen(QPen(QColor(40, 40, 60), 1))
        for i in range(0, self.width(), 50):
            painter.drawLine(i, 0, i, self.height())
            painter.drawLine(0, i, self.width(), i)
        
        # Draw skeleton
        if hasattr(self, 'bones') and self.joint_positions:
            # Bones
            painter.setPen(QPen(QColor(100, 150, 200, 200), 3))
            for start, end in self.bones:
                if start in self.joint_positions and end in self.joint_positions:
                    painter.drawLine(self.joint_positions[start], self.joint_positions[end])
            
            # Joints
            for name, pos in self.joint_positions.items():
                color = QColor(255, 200, 100) if name == "head" else QColor(80, 150, 255)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                painter.drawEllipse(pos, 6, 6)
                
                # Label
                painter.setPen(QColor(200, 200, 200, 150))
                painter.setFont(QFont("Monospace", 6))
                painter.drawText(pos.x() + 8, pos.y() - 4, name[:3])
        
        # Info
        if self.current_pose:
            painter.setPen(QColor(255, 215, 0))
            painter.setFont(QFont("Monospace", 9))
            painter.drawText(10, 25, f"Pose: {self.current_pose.name}")
            painter.drawText(10, 45, f"Type: {self.current_pose.pose_type.value}")
            if self.is_playing and self.current_motion:
                painter.drawText(10, 65, f"Time: {self.animation_time:.2f}/{self.current_motion.duration:.2f}s")
    
    def resizeEvent(self, event):
        self._compute_joint_positions()
        super().resizeEvent(event)


# ============================================================================
# Main CAE GUI
# ============================================================================

class ROCACAEGUI(QMainWindow):
    """Main ROCA-CAE Animation GUI."""
    
    def __init__(self):
        super().__init__()
        
        # Storage
        self.poses: Dict[str, Pose] = {}
        self.motions: Dict[str, Motion] = {}
        self._init_seed_data()
        
        self._init_ui()
        self._update_lists()
        self._welcome()
    
    def _init_seed_data(self):
        """Create example poses."""
        # Default pose
        default_joints = {
            "head": Vector3(0, 0.9, 0), "neck": Vector3(0, 0.75, 0),
            "chest": Vector3(0, 0.6, 0), "spine": Vector3(0, 0.4, 0),
            "hip": Vector3(0, 0.2, 0), "root": Vector3(0, 0, 0),
            "shoulder_L": Vector3(-0.3, 0.55, 0), "elbow_L": Vector3(-0.5, 0.5, 0),
            "hand_L": Vector3(-0.6, 0.45, 0),
            "shoulder_R": Vector3(0.3, 0.55, 0), "elbow_R": Vector3(0.5, 0.5, 0),
            "hand_R": Vector3(0.6, 0.45, 0),
            "hip_L": Vector3(-0.25, 0.15, 0), "knee_L": Vector3(-0.3, -0.1, 0),
            "foot_L": Vector3(-0.35, -0.3, 0),
            "hip_R": Vector3(0.25, 0.15, 0), "knee_R": Vector3(0.3, -0.1, 0),
            "foot_R": Vector3(0.35, -0.3, 0),
        }
        
        idle_pose = Pose(
            id="pose_idle", name="Idle", pose_type=PoseType.IDLE,
            joints=default_joints, duration=2.0
        )
        self.poses["pose_idle"] = idle_pose
        
        # Walk pose (slightly modified)
        walk_joints = default_joints.copy()
        walk_joints["hip_L"] = Vector3(-0.3, 0.1, 0.1)
        walk_joints["hip_R"] = Vector3(0.2, 0.1, -0.1)
        walk_pose = Pose(
            id="pose_walk", name="Walk", pose_type=PoseType.WALK,
            joints=walk_joints, duration=1.0
        )
        self.poses["pose_walk"] = walk_pose
        
        # Run pose
        run_joints = default_joints.copy()
        run_joints["hip_L"] = Vector3(-0.35, 0.05, 0.2)
        run_joints["hip_R"] = Vector3(0.25, 0.05, -0.2)
        run_joints["elbow_L"] = Vector3(-0.55, 0.45, -0.1)
        run_joints["elbow_R"] = Vector3(0.55, 0.45, 0.1)
        run_pose = Pose(
            id="pose_run", name="Run", pose_type=PoseType.RUN,
            joints=run_joints, duration=0.8
        )
        self.poses["pose_run"] = run_pose
    
    def _init_ui(self):
        self.setWindowTitle("ROCA-CAE - Character Animation Studio")
        self.setGeometry(50, 50, 1400, 850)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a1a; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', monospace; }
            QSplitter::handle { background-color: #333; width: 2px; }
            QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 12px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #0f3460; border: 1px solid #1a5276;
                border-radius: 4px; padding: 6px 12px;
            }
            QPushButton:hover { background-color: #1a5276; }
            QLineEdit, QTextEdit { background-color: #0f0f23; border: 1px solid #333; border-radius: 4px; }
            QListWidget { background-color: #1a1a2e; border: 1px solid #333; border-radius: 4px; }
            QComboBox { background-color: #0f0f23; border: 1px solid #333; border-radius: 4px; padding: 4px; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT PANEL - Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        
        # Pose Creation
        pose_group = QGroupBox("Create Pose")
        pose_layout = QGridLayout(pose_group)
        
        pose_layout.addWidget(QLabel("Name:"), 0, 0)
        self.pose_name = QTextEdit()
        self.pose_name.setMaximumHeight(30)
        pose_layout.addWidget(self.pose_name, 0, 1)
        
        pose_layout.addWidget(QLabel("Type:"), 1, 0)
        self.pose_type = QComboBox()
        self.pose_type.addItems([t.value for t in PoseType])
        pose_layout.addWidget(self.pose_type, 1, 1)
        
        pose_layout.addWidget(QLabel("Duration:"), 2, 0)
        self.pose_duration = QDoubleSpinBox()
        self.pose_duration.setRange(0.1, 10.0)
        self.pose_duration.setValue(1.0)
        pose_layout.addWidget(self.pose_duration, 2, 1)
        
        create_pose_btn = QPushButton("Create Pose")
        create_pose_btn.clicked.connect(self._create_pose)
        pose_layout.addWidget(create_pose_btn, 3, 0, 1, 2)
        
        left_layout.addWidget(pose_group)
        
        # Motion Creation
        motion_group = QGroupBox("Create Motion")
        motion_layout = QGridLayout(motion_group)
        
        motion_layout.addWidget(QLabel("Name:"), 0, 0)
        self.motion_name = QTextEdit()
        self.motion_name.setMaximumHeight(30)
        motion_layout.addWidget(self.motion_name, 0, 1)
        
        motion_layout.addWidget(QLabel("Type:"), 1, 0)
        self.motion_type = QComboBox()
        self.motion_type.addItems([t.value for t in MotionType])
        motion_layout.addWidget(self.motion_type, 1, 1)
        
        motion_layout.addWidget(QLabel("Duration:"), 2, 0)
        self.motion_duration = QDoubleSpinBox()
        self.motion_duration.setRange(0.5, 30.0)
        self.motion_duration.setValue(2.0)
        motion_layout.addWidget(self.motion_duration, 2, 1)
        
        self.loop_cb = QCheckBox("Loop")
        motion_layout.addWidget(self.loop_cb, 3, 0, 1, 2)
        
        create_motion_btn = QPushButton("Create Motion")
        create_motion_btn.clicked.connect(self._create_motion)
        motion_layout.addWidget(create_motion_btn, 4, 0, 1, 2)
        
        left_layout.addWidget(motion_group)
        
        # Interpolation
        interp_group = QGroupBox("Pose Interpolation")
        interp_layout = QGridLayout(interp_group)
        
        interp_layout.addWidget(QLabel("From:"), 0, 0)
        self.from_pose = QComboBox()
        interp_layout.addWidget(self.from_pose, 0, 1)
        
        interp_layout.addWidget(QLabel("To:"), 1, 0)
        self.to_pose = QComboBox()
        interp_layout.addWidget(self.to_pose, 1, 1)
        
        interp_layout.addWidget(QLabel("Blend:"), 2, 0)
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_label = QLabel("0.50")
        interp_layout.addWidget(self.blend_slider, 2, 1)
        interp_layout.addWidget(self.blend_label, 3, 0, 1, 2)
        self.blend_slider.valueChanged.connect(lambda v: self.blend_label.setText(f"{v/100:.2f}"))
        
        interp_btn = QPushButton("Interpolate & Create")
        interp_btn.clicked.connect(self._interpolate)
        interp_layout.addWidget(interp_btn, 4, 0, 1, 2)
        
        left_layout.addWidget(interp_group)
        
        # Lists
        list_group = QGroupBox("Assets")
        list_layout = QVBoxLayout(list_group)
        
        self.pose_list = QListWidget()
        self.pose_list.itemClicked.connect(self._on_pose_selected)
        list_layout.addWidget(QLabel("Poses:"))
        list_layout.addWidget(self.pose_list)
        
        self.motion_list = QListWidget()
        self.motion_list.itemClicked.connect(self._on_motion_selected)
        list_layout.addWidget(QLabel("Motions:"))
        list_layout.addWidget(self.motion_list)
        
        left_layout.addWidget(list_group)
        
        splitter.addWidget(left)
        
        # RIGHT PANEL - Visualization
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.viz = PoseVisualizer()
        right_layout.addWidget(self.viz, 3)
        
        # Playback controls
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #16213e; border-radius: 6px;")
        control_layout = QHBoxLayout(control_frame)
        
        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(self._play_motion)
        stop_btn = QPushButton("■ Stop")
        stop_btn.clicked.connect(self._stop_motion)
        
        control_layout.addWidget(play_btn)
        control_layout.addWidget(stop_btn)
        control_layout.addStretch()
        
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 2.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.25)
        control_layout.addWidget(self.speed_spin)
        
        right_layout.addWidget(control_frame)
        
        splitter.addWidget(right)
        splitter.setSizes([450, 950])
        layout.addWidget(splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("ROCA-CAE Ready. Create poses and motions!")
    
    def _create_pose(self):
        name = self.pose_name.toPlainText().strip()
        if not name:
            name = f"pose_{len(self.poses) + 1}"
        
        # Use default joint structure from idle pose
        base_pose = self.poses.get("pose_idle")
        if base_pose:
            joints = base_pose.joints.copy()
        else:
            joints = {}
        
        pose = Pose(
            id=f"pose_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            pose_type=PoseType(self.pose_type.currentText()),
            joints=joints,
            duration=self.pose_duration.value()
        )
        
        self.poses[pose.id] = pose
        self._update_lists()
        self.viz.set_pose(pose)
        self.pose_name.clear()
        self.status_bar.showMessage(f"Created pose: {name}")
    
    def _create_motion(self):
        name = self.motion_name.toPlainText().strip()
        if not name:
            name = f"motion_{len(self.motions) + 1}"
        
        # Use all poses as keyframes
        poses = list(self.poses.values())
        if not poses:
            self.status_bar.showMessage("Create some poses first!")
            return
        
        motion = Motion(
            id=f"motion_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            motion_type=MotionType(self.motion_type.currentText()),
            poses=poses,
            duration=self.motion_duration.value(),
            loop=self.loop_cb.isChecked()
        )
        
        self.motions[motion.id] = motion
        self._update_lists()
        self.motion_name.clear()
        self.status_bar.showMessage(f"Created motion: {name} with {len(poses)} poses")
    
    def _interpolate(self):
        from_id = self.from_pose.currentData()
        to_id = self.to_pose.currentData()
        
        if not from_id or not to_id or from_id == to_id:
            self.status_bar.showMessage("Select two different poses")
            return
        
        pose_a = self.poses.get(from_id)
        pose_b = self.poses.get(to_id)
        
        if not pose_a or not pose_b:
            return
        
        t = self.blend_slider.value() / 100.0
        new_pose = pose_a.interpolate_to(pose_b, t)
        new_pose.id = f"interp_{pose_a.id}_{pose_b.id}_{int(t*100)}"
        new_pose.name = f"{pose_a.name}→{pose_b.name} ({t:.0%})"
        
        self.poses[new_pose.id] = new_pose
        self._update_lists()
        self.viz.set_pose(new_pose)
        self.status_bar.showMessage(f"Created interpolated pose: {new_pose.name}")
    
    def _on_pose_selected(self, item):
        pose_id = item.data(Qt.ItemDataRole.UserRole)
        pose = self.poses.get(pose_id)
        if pose:
            self.viz.set_pose(pose)
            self.viz.stop()
            self.status_bar.showMessage(f"Selected pose: {pose.name}")
    
    def _on_motion_selected(self, item):
        motion_id = item.data(Qt.ItemDataRole.UserRole)
        motion = self.motions.get(motion_id)
        if motion:
            self.viz.set_motion(motion)
            self.status_bar.showMessage(f"Selected motion: {motion.name}")
    
    def _play_motion(self):
        if self.viz.current_motion:
            # Update speed
            self.viz.anim_timer.start(int(16 / self.speed_spin.value()))
            self.viz.play()
    
    def _stop_motion(self):
        self.viz.stop()
    
    def _update_lists(self):
        # Update pose list
        self.pose_list.clear()
        self.from_pose.clear()
        self.to_pose.clear()
        
        for pose in self.poses.values():
            item = QListWidgetItem(f"🧍 {pose.name} ({pose.pose_type.value})")
            item.setData(Qt.ItemDataRole.UserRole, pose.id)
            self.pose_list.addItem(item)
            
            self.from_pose.addItem(pose.name, pose.id)
            self.to_pose.addItem(pose.name, pose.id)
        
        # Update motion list
        self.motion_list.clear()
        for motion in self.motions.values():
            loop_str = " 🔄" if motion.loop else ""
            item = QListWidgetItem(f"🎬 {motion.name} ({motion.motion_type.value}){loop_str}")
            item.setData(Qt.ItemDataRole.UserRole, motion.id)
            self.motion_list.addItem(item)
    
    def _welcome(self):
        self.status_bar.showMessage(f"ROCA-CAE Ready | {len(self.poses)} poses, {len(self.motions)} motions")


# ============================================================================
# Add missing QCheckBox import
# ============================================================================

from PyQt6.QtWidgets import QCheckBox


# ============================================================================
# Main Entry
# ============================================================================

def main():
    print("=" * 60)
    print("ROCA-CAE - Character Animation Studio")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    app.setApplicationName("ROCA-CAE")
    
    window = ROCACAEGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()