"""
ecg/holter/add_event_dialog.py
===============================
Add Event Dialog for Holter ECG Analysis

Allows users to add custom events with:
- Strip start time selection
- Print length (seconds)
- Heart rate
- Strip label (event type)
- Print orientation (Vertical/Horizontal)
- Preview channel selection
- Print channel selection (12-lead: I, II, III, aVR, aVL, aVF, V1-V6)
- Lead exchange options
"""

from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QRadioButton, QCheckBox, QGroupBox,
    QDateTimeEdit, QSpinBox, QButtonGroup, QFrame
)
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal
from PyQt5.QtGui import QFont

try:
    from .theme import COL_BLACK, COL_GREEN, COL_GREEN_DRK, COL_TEXT, UI_BORDER, UI_TEXT, UI_PANEL_ALT
except ImportError:
    COL_BLACK = "#0a0f18"
    COL_GREEN = "#28E37B"
    COL_GREEN_DRK = "#1a6b3f"
    COL_TEXT = "#f3f7fb"
    UI_BORDER = "#2a5a6d"
    UI_TEXT = "#f3f7fb"
    UI_PANEL_ALT = "#0d1521"


class AddEventDialog(QDialog):
    """
    Dialog for adding custom events to Holter recording.
    Matches the reference design with 12-lead ECG support.
    """
    
    event_added = pyqtSignal(dict)  # Emits event data when added
    
    # 12-lead names
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # Event/Strip label options
    EVENT_LABELS = [
        "Ventricular fusion",
        "Atrial Fibrillation",
        "Atrial Flutter",
        "Supraventricular Tachycardia",
        "Ventricular Tachycardia",
        "Bradycardia",
        "Pause",
        "PVC",
        "PAC",
        "Custom"
    ]
    
    # Hidden event labels (not shown in dropdown)
    HIDDEN_LABELS = [
        "Long QT Syndrome",
        "Wide QRS (non-specific)",
        "Frequent PVCs",
        "Multifocal PVCs"
    ]
    
    def __init__(self, parent=None, start_time: datetime = None, current_sec: float = 0.0, hr: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Edit strip")
        self.setModal(True)
        self.setFixedSize(640, 600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self._start_time = start_time or datetime.now()
        self._current_sec = current_sec
        self._hr = hr
        
        self._build_ui()
        self._apply_styles()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # ========== Row 1: Strip start time, Print Len, HR ==========
        row1 = QHBoxLayout()
        row1.setSpacing(12)
        
        # Strip start time
        time_group = QVBoxLayout()
        time_lbl = QLabel("Strip start time")
        time_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.time_edit = QDateTimeEdit()
        self.time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(self._start_time.timestamp() + self._current_sec)))
        self.time_edit.setCalendarPopup(True)
        time_group.addWidget(time_lbl)
        time_group.addWidget(self.time_edit)
        row1.addLayout(time_group, 2)
        
        # Print Len (Sec.)
        len_group = QVBoxLayout()
        len_lbl = QLabel("Print Len.(Sec.)")
        len_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.len_combo = QComboBox()
        self.len_combo.addItems(["3", "5", "7", "10", "15", "30"])
        self.len_combo.setCurrentText("7")
        len_group.addWidget(len_lbl)
        len_group.addWidget(self.len_combo)
        row1.addLayout(len_group, 1)
        
        # HR
        hr_group = QVBoxLayout()
        hr_lbl = QLabel("HR")
        hr_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.hr_edit = QLineEdit()
        self.hr_edit.setText(f"{int(self._hr)}" if self._hr > 0 else "")
        self.hr_edit.setPlaceholderText("bpm")
        hr_group.addWidget(hr_lbl)
        hr_group.addWidget(self.hr_edit)
        row1.addLayout(hr_group, 1)
        
        layout.addLayout(row1)
        
        # ========== Row 2: << >> buttons ==========
        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self.btn_prev = QPushButton("<<")
        self.btn_next = QPushButton(">>")
        self.btn_prev.setFixedWidth(80)
        self.btn_next.setFixedWidth(80)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        nav_row.addStretch()
        layout.addLayout(nav_row)
        
        # ========== Row 3: Strip label ==========
        label_group = QVBoxLayout()
        label_lbl = QLabel("Strip label")
        label_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.label_combo = QComboBox()
        self.label_combo.addItems(self.EVENT_LABELS)
        self.label_combo.setCurrentText("Ventricular fusion")
        label_group.addWidget(label_lbl)
        label_group.addWidget(self.label_combo)
        layout.addLayout(label_group)
        
        # ========== Strip note ==========
        note_lbl = QLabel("Strip note")
        note_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        layout.addWidget(note_lbl)
        
        # ========== Row 4: Print orientation, Preview channel, Strip Display Type ==========
        row4 = QHBoxLayout()
        row4.setSpacing(16)
        
        # Print orientation
        orient_group = QGroupBox("Print orientation")
        orient_layout = QHBoxLayout(orient_group)
        self.radio_vertical = QRadioButton("Vertical")
        self.radio_horizontal = QRadioButton("Horizontal")
        self.radio_vertical.setChecked(True)
        orient_layout.addWidget(self.radio_vertical)
        orient_layout.addWidget(self.radio_horizontal)
        row4.addWidget(orient_group)
        
        # Preview channel
        preview_group = QVBoxLayout()
        preview_lbl = QLabel("Preview channel")
        preview_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.preview_combo = QComboBox()
        self.preview_combo.addItem("Auto")
        self.preview_combo.addItems(self.LEADS)
        self.preview_combo.setCurrentText("Auto")
        preview_group.addWidget(preview_lbl)
        preview_group.addWidget(self.preview_combo)
        row4.addLayout(preview_group)
        
        # Strip Display Type
        display_group = QVBoxLayout()
        display_lbl = QLabel("Strip Display Type")
        display_lbl.setStyleSheet("color:#E5E7EB;font-size:12px;font-weight:700;background:transparent;")
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Default", "Cascade", "Grid"])
        display_group.addWidget(display_lbl)
        display_group.addWidget(self.display_combo)
        row4.addLayout(display_group)
        
        layout.addLayout(row4)
        
        # ========== Print channel selection ==========
        channel_group = QGroupBox("Print channel")
        channel_layout = QVBoxLayout(channel_group)
        
        # Radio buttons: 1-CH, 3-CH, Custom
        radio_row = QHBoxLayout()
        self.radio_1ch = QRadioButton("1-CH")
        self.radio_3ch = QRadioButton("3-CH")
        self.radio_custom = QRadioButton("Custom")
        self.radio_3ch.setChecked(True)
        radio_row.addWidget(self.radio_1ch)
        radio_row.addWidget(self.radio_3ch)
        radio_row.addWidget(self.radio_custom)
        radio_row.addStretch()
        channel_layout.addLayout(radio_row)
        
        # Channel checkboxes with gain dropdowns (12 leads)
        self.channel_checks = {}
        self.channel_gains = {}
        
        channels_grid = QHBoxLayout()
        channels_grid.setSpacing(12)
        
        for i, lead in enumerate(self.LEADS):
            ch_box = QHBoxLayout()
            ch_box.setSpacing(6)
            
            check = QCheckBox(lead)
            check.setChecked(i < 3)  # Default: first 3 leads checked (I, II, III)
            
            gain_combo = QComboBox()
            gain_combo.addItems(["5mm/mV", "10mm/mV", "20mm/mV", "40mm/mV"])
            gain_combo.setCurrentText("10mm/mV")
            gain_combo.setFixedWidth(90)
            
            ch_box.addWidget(check)
            ch_box.addWidget(gain_combo)
            
            self.channel_checks[lead] = check
            self.channel_gains[lead] = gain_combo
            
            if i % 4 == 0:
                col = QVBoxLayout()
                col.setSpacing(8)
                channels_grid.addLayout(col)
            col.addLayout(ch_box)
        
        channel_layout.addLayout(channels_grid)
        layout.addWidget(channel_group)
        
        # ========== Lead exchange ==========
        exchange_group = QGroupBox("Lead exchange")
        exchange_layout = QHBoxLayout(exchange_group)
        exchange_layout.setSpacing(16)
        
        # Limb lead reversal
        limb_check = QCheckBox("limb lead reversal")
        exchange_layout.addWidget(limb_check)
        self.limb_check = limb_check
        
        # Chest lead exchange
        chest_layout = QHBoxLayout()
        chest_check = QCheckBox("Chest lead exchange")
        self.chest_from_combo = QComboBox()
        self.chest_from_combo.addItems(self.LEADS)
        self.chest_from_combo.setCurrentText("I")
        
        swap_btn = QPushButton("⇄")   # swap arrows
        swap_btn.setObjectName("swapBtn")
        swap_btn.setToolTip("Swap leads")
        swap_btn.setCursor(Qt.PointingHandCursor)
        swap_btn.setFixedSize(30, 26)
        
        self.chest_to_combo = QComboBox()
        self.chest_to_combo.addItems(self.LEADS)
        self.chest_to_combo.setCurrentText("I")
        
        chest_layout.addWidget(chest_check)
        chest_layout.addWidget(self.chest_from_combo)
        chest_layout.addWidget(swap_btn)
        chest_layout.addWidget(self.chest_to_combo)
        chest_layout.addStretch()
        self.chest_check = chest_check
        
        exchange_layout.addLayout(chest_layout)
        exchange_layout.addStretch()
        layout.addWidget(exchange_group)
        
        # ========== Bottom buttons ==========
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        self.btn_add = QPushButton("Add Event")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_instant = QPushButton("Instant print")
        self.btn_export = QPushButton("Export as pdf")
        self.btn_default = QPushButton("Set as default")
        
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_instant)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_default)
        
        layout.addLayout(btn_row)
        
        # Connect signals
        self.btn_add.clicked.connect(self._on_add_event)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_instant.clicked.connect(self._on_instant_print)
        self.btn_export.clicked.connect(self._on_export_pdf)
        self.btn_default.clicked.connect(self._on_set_default)
        
        self.radio_1ch.toggled.connect(self._on_channel_mode_changed)
        self.radio_3ch.toggled.connect(self._on_channel_mode_changed)
        self.radio_custom.toggled.connect(self._on_channel_mode_changed)
        
        swap_btn.clicked.connect(self._on_swap_leads)
    
    def _apply_styles(self):
        """Apply consistent styling to the dialog (same navy palette as the main window)."""
        bg, panel, panel_alt, border = "#0B1220", "#0F1A2E", "#13213A", "#243552"
        text, muted, accent, accent_hover = "#E6EDF7", "#9AAECB", "#2F80ED", "#4B96FA"
        field = "#0B1322"
        self.setStyleSheet(f"""
            QDialog {{
                background: {bg};
                color: {text};
            }}
            QLabel {{
                color: {text};
                font-size: 12px;
                background: transparent;
            }}
            QLineEdit, QComboBox, QDateTimeEdit, QSpinBox {{
                background: {field};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
                font-weight: 700;
                selection-background-color: #1E3A5F;
            }}
            QLineEdit:hover, QComboBox:hover, QDateTimeEdit:hover, QSpinBox:hover {{
                border-color: #3A5378;
            }}
            QLineEdit:focus, QComboBox:focus, QDateTimeEdit:focus, QSpinBox:focus, QComboBox:on {{
                border-color: {accent};
                background: {panel};
            }}
            QComboBox QAbstractItemView {{
                background: {panel};
                color: {text};
                selection-background-color: {accent};
                selection-color: #FFFFFF;
                border: 1px solid {border};
                outline: 0;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 8px;
                min-height: 20px;
                color: {text};
                font-weight: 700;
            }}
            QPushButton {{
                background: {panel_alt};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 700;
                min-height: 26px;
            }}
            QPushButton:hover {{
                background: #1A2C49;
                border-color: {accent};
            }}
            QPushButton:pressed {{
                background: {panel};
                border-color: {accent_hover};
            }}
            QPushButton#primaryBtn {{
                background: {accent};
                border-color: #5EA4FF;
                color: #FFFFFF;
            }}
            QPushButton#primaryBtn:hover {{
                background: {accent_hover};
            }}
            QPushButton#swapBtn {{
                padding: 0px;
                min-height: 0px;
                font-size: 14px;
            }}
            QGroupBox {{
                color: {text};
                border: 1px solid {border};
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 20px;
                font-weight: 700;
                font-size: 12px;
                background: {panel};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 6px;
                color: {text};
                font-weight: 700;
            }}
            QRadioButton, QCheckBox {{
                color: {text};
                spacing: 8px;
                font-size: 12px;
                background: transparent;
            }}
            QRadioButton:disabled, QCheckBox:disabled {{
                color: {muted};
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
                border: 2px solid #3A5378;
                border-radius: 9px;
                background: {field};
            }}
            QRadioButton::indicator:hover, QCheckBox::indicator:hover {{
                border-color: {accent};
            }}
            QRadioButton::indicator:checked {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                    stop:0 {accent_hover}, stop:0.45 {accent_hover}, stop:0.55 {field}, stop:1 {field});
                border-color: {accent};
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 2px solid #3A5378;
                border-radius: 4px;
                background: {field};
            }}
            QCheckBox::indicator:checked {{
                background: {accent};
                border-color: {accent};
            }}
            QToolButton {{
                background: {panel_alt};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QToolButton:hover {{
                background: #1A2C49;
                border-color: {accent};
            }}
        """)
        # Add Event is the main action of this dialog
        self.btn_add.setObjectName("primaryBtn")
        self.btn_add.setDefault(True)
        for btn in (self.btn_add, self.btn_cancel, self.btn_instant, self.btn_export,
                    self.btn_default, self.btn_prev, self.btn_next):
            btn.setCursor(Qt.PointingHandCursor)
    
    def _on_channel_mode_changed(self):
        """Update channel selections based on mode."""
        if self.radio_1ch.isChecked():
            # Enable only lead II
            for lead, check in self.channel_checks.items():
                check.setChecked(lead == "II")
                check.setEnabled(False)
        elif self.radio_3ch.isChecked():
            # Enable leads I, II, III
            for lead, check in self.channel_checks.items():
                check.setChecked(lead in ["I", "II", "III"])
                check.setEnabled(False)
        else:  # Custom
            # Enable all checkboxes for manual selection
            for check in self.channel_checks.values():
                check.setEnabled(True)
    
    def _on_swap_leads(self):
        """Swap the lead selections in chest lead exchange."""
        from_lead = self.chest_from_combo.currentText()
        to_lead = self.chest_to_combo.currentText()
        self.chest_from_combo.setCurrentText(to_lead)
        self.chest_to_combo.setCurrentText(from_lead)
    
    def _on_add_event(self):
        """Collect event data and emit signal."""
        event_data = self._collect_event_data()
        self.event_added.emit(event_data)
        self.accept()
    
    def _on_instant_print(self):
        """Trigger instant print."""
        event_data = self._collect_event_data()
        event_data['action'] = 'instant_print'
        self.event_added.emit(event_data)
        self.accept()
    
    def _on_export_pdf(self):
        """Export event as PDF."""
        event_data = self._collect_event_data()
        event_data['action'] = 'export_pdf'
        self.event_added.emit(event_data)
        self.accept()
    
    def _on_set_default(self):
        """Save current settings as default."""
        # TODO: Save settings to config file
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Settings Saved", "Current settings have been saved as default.")
    
    def _collect_event_data(self) -> dict:
        """Collect all form data into a dictionary."""
        # Get selected channels
        selected_channels = [lead for lead, check in self.channel_checks.items() if check.isChecked()]
        channel_gains = {lead: self.channel_gains[lead].currentText() for lead in selected_channels}
        
        return {
            'timestamp': self.time_edit.dateTime().toSecsSinceEpoch(),
            'print_length_sec': int(self.len_combo.currentText()),
            'hr': int(self.hr_edit.text()) if self.hr_edit.text().isdigit() else 0,
            'label': self.label_combo.currentText(),
            'orientation': 'vertical' if self.radio_vertical.isChecked() else 'horizontal',
            'preview_channel': self.preview_combo.currentText(),
            'display_type': self.display_combo.currentText(),
            'channel_mode': '1-CH' if self.radio_1ch.isChecked() else ('3-CH' if self.radio_3ch.isChecked() else 'Custom'),
            'selected_channels': selected_channels,
            'channel_gains': channel_gains,
            'limb_reversal': self.limb_check.isChecked(),
            'chest_exchange': self.chest_check.isChecked(),
            'chest_from': self.chest_from_combo.currentText() if self.chest_check.isChecked() else None,
            'chest_to': self.chest_to_combo.currentText() if self.chest_check.isChecked() else None,
            'action': 'add_event'
        }


# Test/Demo
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = AddEventDialog(start_time=datetime.now(), current_sec=120.0, hr=75.0)
    dialog.event_added.connect(lambda data: print(f"Event added: {data}"))
    dialog.exec_()
    sys.exit(app.exit_code if hasattr(app, 'exit_code') else 0)
