from __future__ import annotations

import os
from typing import Dict, Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QStackedWidget, QGroupBox, QRadioButton, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QFrame, QGridLayout, QScrollArea, QMessageBox,
    QButtonGroup
)

from ..holter_helpers import (
    UI_ACCENT, UI_ACCENT_HOVER, UI_BG, UI_BORDER, UI_CARD, UI_MUTED,
    UI_PANEL, UI_PANEL_ALT, UI_SUCCESS, UI_TEXT, UI_WARNING, _style_btn, _style_active_btn
)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ALL_ROWS = LEADS + ["All"]
GAIN_VALUES = ["5", "10", "20"]


class ClickableRadioButton(QRadioButton):
    """QRadioButton that allows unchecking when clicked while already checked."""
    def mousePressEvent(self, event):
        if self.isChecked():
            # If already checked and clicked, uncheck it manually
            group = self.group()
            if group:
                group.setExclusive(False)
            self.setChecked(False)
            if group:
                group.setExclusive(True)
        else:
            super().mousePressEvent(event)


class HolterRecordSettingsPanel(QWidget):
    """
    Comprehensive Record Settings Panel containing 3 main option categories:
    1. Display setting
    2. Clinical setting
    3. HRV Setting
    """
    settings_changed = pyqtSignal(dict)

    def __init__(self, duration_hours: float = 24.0, session_name: str = "Current session", parent=None):
        super().__init__(parent)
        self._duration_hours = duration_hours
        self._session_name = session_name
        self.setStyleSheet(f"background:{UI_BG}; color:{UI_TEXT};")
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Info Banner
        header_card = QFrame()
        header_card.setStyleSheet(
            f"QFrame {{ background: {UI_CARD}; border: 1px solid {UI_BORDER}; border-radius: 8px; padding: 10px; }}"
        )
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title_lbl = QLabel("Record Settings")
        title_lbl.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {UI_TEXT}; border: none;")
        header_layout.addWidget(title_lbl)

        header_layout.addStretch(1)

        info_lbl = QLabel(f"Recording duration: {self._duration_hours:g} hours | Session: {self._session_name}")
        info_lbl.setStyleSheet(f"font-size: 12px; color: {UI_MUTED}; border: none;")
        header_layout.addWidget(info_lbl)

        main_layout.addWidget(header_card)

        # Settings Body with Left Sidebar + Right Stack
        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        # Left Sidebar (Exactly 3 Options: Display, Clinical, HRV)
        self._sidebar_list = QListWidget()
        self._sidebar_list.setFixedWidth(200)
        self._sidebar_list.setFocusPolicy(Qt.NoFocus)
        self._sidebar_list.setStyleSheet(f"""
            QListWidget {{
                background: {UI_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
                outline: none;
                padding: 6px;
            }}
            QListWidget::item {{
                height: 42px;
                color: {UI_MUTED};
                font-size: 13px;
                font-weight: 600;
                border-radius: 6px;
                padding-left: 12px;
                margin-bottom: 4px;
            }}
            QListWidget::item:hover {{
                background: {UI_PANEL_ALT};
                color: {UI_TEXT};
            }}
            QListWidget::item:selected {{
                background: {UI_ACCENT};
                color: #FFFFFF;
                font-weight: 700;
            }}
        """)

        item_display = QListWidgetItem("Display setting")
        item_clinical = QListWidgetItem("Clinical setting")
        item_hrv = QListWidgetItem("HRV Setting")

        self._sidebar_list.addItem(item_display)
        self._sidebar_list.addItem(item_clinical)
        self._sidebar_list.addItem(item_hrv)

        body_layout.addWidget(self._sidebar_list)

        # Right Stacked Widget
        self._stacked_widget = QStackedWidget()
        self._stacked_widget.setStyleSheet(f"""
            QStackedWidget {{
                background: {UI_CARD};
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
            }}
        """)

        # Create pages
        self._page_display = self._create_display_settings_page()
        self._page_clinical = self._create_clinical_settings_page()
        self._page_hrv = self._create_hrv_settings_page()

        self._stacked_widget.addWidget(self._page_display)
        self._stacked_widget.addWidget(self._page_clinical)
        self._stacked_widget.addWidget(self._page_hrv)

        body_layout.addWidget(self._stacked_widget, 1)
        main_layout.addLayout(body_layout, 1)

        # Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(10)

        self._btn_wave_grid = QPushButton("Wave/Grid setting")
        self._btn_wave_grid.setStyleSheet(_style_btn())
        self._btn_wave_grid.setFixedHeight(34)
        bottom_bar.addWidget(self._btn_wave_grid)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setStyleSheet(_style_btn())
        self._btn_reset.setFixedHeight(34)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        bottom_bar.addWidget(self._btn_reset)

        bottom_bar.addStretch(1)

        self._btn_ok = QPushButton("OK")
        self._btn_ok.setStyleSheet(_style_active_btn())
        self._btn_ok.setFixedWidth(90)
        self._btn_ok.setFixedHeight(34)
        self._btn_ok.clicked.connect(self._on_ok_clicked)
        bottom_bar.addWidget(self._btn_ok)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setStyleSheet(_style_btn())
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.setFixedHeight(34)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        bottom_bar.addWidget(self._btn_cancel)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setStyleSheet(_style_btn())
        self._btn_apply.setFixedWidth(90)
        self._btn_apply.setFixedHeight(34)
        self._btn_apply.clicked.connect(self._on_apply_clicked)
        bottom_bar.addWidget(self._btn_apply)

        main_layout.addLayout(bottom_bar)

        # Connect navigation
        self._sidebar_list.currentRowChanged.connect(self._stacked_widget.setCurrentIndex)
        self._sidebar_list.setCurrentRow(0)

    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                font-weight: 700;
                font-size: 13px;
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                background: {UI_PANEL};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: {UI_ACCENT_HOVER};
            }}
        """

    def _create_display_settings_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # 1. Lead gain adjust Group (Renamed from Channel gain adjust)
        group_gain = QGroupBox("Lead gain adjust")
        group_gain.setStyleSheet(self._group_style())
        grid_gain = QGridLayout(group_gain)
        grid_gain.setContentsMargins(14, 16, 14, 14)
        grid_gain.setHorizontalSpacing(24)
        grid_gain.setVerticalSpacing(10)

        self._gain_radios = {}
        self._invert_checks = {}

        # Column headers
        lbl_lead_hdr = QLabel("Lead")
        lbl_lead_hdr.setStyleSheet(f"font-weight: 700; color: {UI_MUTED}; font-size: 12px;")
        grid_gain.addWidget(lbl_lead_hdr, 0, 0)

        for col, val in enumerate(GAIN_VALUES):
            lbl_val_hdr = QLabel(f"{val} mm/mV")
            lbl_val_hdr.setStyleSheet(f"font-weight: 700; color: {UI_MUTED}; font-size: 12px;")
            grid_gain.addWidget(lbl_val_hdr, 0, col + 1)

        lbl_inv_hdr = QLabel("Invert")
        lbl_inv_hdr.setStyleSheet(f"font-weight: 700; color: {UI_MUTED}; font-size: 12px;")
        grid_gain.addWidget(lbl_inv_hdr, 0, len(GAIN_VALUES) + 1)

        # Rows for I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, All
        self._gain_btn_groups = {}  # per-row QButtonGroup for correct mutual exclusion
        for row_idx, lead in enumerate(ALL_ROWS, start=1):
            lbl_lead = QLabel(lead)
            lbl_lead.setStyleSheet(f"font-weight: 700; color: {UI_TEXT}; font-size: 12px;")
            grid_gain.addWidget(lbl_lead, row_idx, 0)

            btn_grp = QButtonGroup(self)  # exclusive within this row only
            btn_grp.setExclusive(True)
            self._gain_btn_groups[lead] = btn_grp

            self._gain_radios[lead] = []
            for col_idx, val in enumerate(GAIN_VALUES):
                r = ClickableRadioButton(val)
                r.setAutoExclusive(False)   # managed by QButtonGroup instead
                r.setStyleSheet(f"color: {UI_TEXT}; font-size: 12px;")
                # Default: 10 mm/mV (index 1) for all 12 leads; nothing for All
                if lead != "All" and val == "10":
                    r.setChecked(True)
                btn_grp.addButton(r, col_idx)
                grid_gain.addWidget(r, row_idx, col_idx + 1)
                self._gain_radios[lead].append(r)

            chk_invert = QCheckBox("")
            chk_invert.setStyleSheet(f"color: {UI_TEXT}; font-size: 12px;")
            grid_gain.addWidget(chk_invert, row_idx, len(GAIN_VALUES) + 1)
            self._invert_checks[lead] = chk_invert

        # --- "All" row logic ---
        # Gain: when user picks any gain in All row → clear individual lead selections
        for col_idx, r in enumerate(self._gain_radios["All"]):
            r.toggled.connect(lambda checked, idx=col_idx: self._on_all_gain_toggled(checked, idx))

        # Individual lead gain: when user picks gain in any lead row → deselect All row gain options
        for lead in LEADS:
            for r in self._gain_radios[lead]:
                r.toggled.connect(self._on_individual_gain_toggled)

        # Invert: when user ticks/unticks All invert → uncheck individual lead invert checkboxes
        self._invert_checks["All"].toggled.connect(self._on_all_invert_toggled)

        # Individual lead invert: when user ticks/unticks any lead invert → uncheck All invert checkbox
        for lead in LEADS:
            self._invert_checks[lead].toggled.connect(self._on_individual_invert_toggled)

        layout.addWidget(group_gain)

        # 2. Full disclosure settings Group
        group_fd = QGroupBox("Full disclosure settings")
        group_fd.setStyleSheet(self._group_style())
        grid_fd = QGridLayout(group_fd)
        grid_fd.setContentsMargins(14, 16, 14, 14)
        grid_fd.setHorizontalSpacing(14)
        grid_fd.setVerticalSpacing(10)

        lbl_width = QLabel("Strip line width")
        lbl_width.setStyleSheet(f"color: {UI_TEXT}; font-size: 12px;")
        grid_fd.addWidget(lbl_width, 0, 0)

        self._combo_width = QComboBox()
        self._combo_width.addItems(["10", "14", "20", "30"])
        self._combo_width.setCurrentText("14")
        self._combo_width.setFixedWidth(80)
        self._combo_width.setStyleSheet(f"""
            QComboBox {{
                background: {UI_PANEL_ALT};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: 4px;
                padding: 3px 8px;
            }}
        """)
        grid_fd.addWidget(self._combo_width, 0, 1)

        lbl_sec = QLabel("s")
        lbl_sec.setStyleSheet(f"color: {UI_MUTED}; font-size: 12px;")
        grid_fd.addWidget(lbl_sec, 0, 2)
        grid_fd.setColumnStretch(3, 1)

        layout.addWidget(group_fd)
        layout.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _on_all_gain_toggled(self, checked: bool, idx: int):
        """When user selects a gain in the 'All' row, deselect options for all individual leads above."""
        if checked:
            for lead in LEADS:
                btn_grp = self._gain_btn_groups[lead]
                btn_grp.setExclusive(False)
                for r in self._gain_radios[lead]:
                    r.blockSignals(True)
                    r.setChecked(False)
                    r.blockSignals(False)
                btn_grp.setExclusive(True)

    def _on_individual_gain_toggled(self, checked: bool):
        """When user selects a gain on an individual lead row, deselect the 'All' row gain radio."""
        if checked:
            btn_grp = self._gain_btn_groups["All"]
            btn_grp.setExclusive(False)
            for r in self._gain_radios["All"]:
                r.blockSignals(True)
                r.setChecked(False)
                r.blockSignals(False)
            btn_grp.setExclusive(True)

    def _on_all_invert_toggled(self, checked: bool):
        """When user ticks 'All' invert checkbox, deselect invert checkboxes for all individual leads above."""
        if checked:
            for lead in LEADS:
                self._invert_checks[lead].blockSignals(True)
                self._invert_checks[lead].setChecked(False)
                self._invert_checks[lead].blockSignals(False)

    def _on_individual_invert_toggled(self, checked: bool):
        """When user ticks/unticks an individual lead's invert checkbox, uncheck the 'All' invert checkbox."""
        if checked:
            self._invert_checks["All"].blockSignals(True)
            self._invert_checks["All"].setChecked(False)
            self._invert_checks["All"].blockSignals(False)

    def _create_clinical_settings_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Section title
        section_title = QLabel("Parameters for arrhythmia analysis")
        section_title.setStyleSheet(
            f"font-size:13px; font-weight:700; color:{UI_ACCENT_HOVER}; "
            f"background:{UI_PANEL_ALT}; border:1px solid {UI_BORDER}; "
            f"border-radius:4px; padding:6px 10px;"
        )
        layout.addWidget(section_title)

        # ---- Left column: basic params + right columns: age-based HR table ----
        body_row = QHBoxLayout()
        body_row.setSpacing(18)

        # ---- LEFT: basic arrhythmia parameters ----
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_grid = QGridLayout(left_widget)
        left_grid.setContentsMargins(0, 0, 0, 0)
        left_grid.setHorizontalSpacing(10)
        left_grid.setVerticalSpacing(10)

        spin_style = (
            f"background:{UI_PANEL_ALT}; color:{UI_TEXT}; "
            f"border:1px solid {UI_BORDER}; border-radius:4px; padding:2px 6px; font-size:12px;"
        )
        lbl_style = f"color:{UI_TEXT}; font-size:12px;"
        hint_style = f"color:{UI_MUTED}; font-size:11px;"

        def _lbl(text):
            w = QLabel(text)
            w.setStyleSheet(lbl_style)
            return w

        def _hint(text):
            w = QLabel(text)
            w.setStyleSheet(hint_style)
            return w

        # Row 0 – Pause duration
        left_grid.addWidget(_lbl("Pause duration ≥"), 0, 0)
        self._spin_pause = QDoubleSpinBox()
        self._spin_pause.setRange(1.0, 5.0)
        self._spin_pause.setValue(2.0)
        self._spin_pause.setSingleStep(0.1)
        self._spin_pause.setDecimals(1)
        self._spin_pause.setFixedWidth(80)
        self._spin_pause.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_pause, 0, 1)
        left_grid.addWidget(_hint("s(1.0~5.0)"), 0, 2)

        # Row 1 – Atrial prematurity
        left_grid.addWidget(_lbl("Atrial prematurity ≥"), 1, 0)
        self._spin_atrial_prem = QDoubleSpinBox()
        self._spin_atrial_prem.setRange(10, 100)
        self._spin_atrial_prem.setValue(20)
        self._spin_atrial_prem.setDecimals(0)
        self._spin_atrial_prem.setFixedWidth(80)
        self._spin_atrial_prem.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_atrial_prem, 1, 1)
        left_grid.addWidget(_hint("%(10~100)"), 1, 2)

        # Row 2 – Atrial Tach. HR
        left_grid.addWidget(_lbl("Atrial Tach. HR ≥"), 2, 0)
        self._spin_atrial_tach_hr = QSpinBox()
        self._spin_atrial_tach_hr.setRange(30, 250)
        self._spin_atrial_tach_hr.setValue(90)
        self._spin_atrial_tach_hr.setFixedWidth(80)
        self._spin_atrial_tach_hr.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_atrial_tach_hr, 2, 1)
        left_grid.addWidget(_hint("bpm(30~250)"), 2, 2)

        # Row 3 – ST Elevation
        left_grid.addWidget(_lbl("ST Elevation ≥"), 3, 0)
        self._spin_st_elev = QDoubleSpinBox()
        self._spin_st_elev.setRange(0.02, 0.5)
        self._spin_st_elev.setValue(0.3)
        self._spin_st_elev.setSingleStep(0.01)
        self._spin_st_elev.setDecimals(2)
        self._spin_st_elev.setFixedWidth(80)
        self._spin_st_elev.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_st_elev, 3, 1)
        left_grid.addWidget(_hint("mV(0.02~0.5)"), 3, 2)

        # Row 4 – ST Depression
        left_grid.addWidget(_lbl("ST Depression ≥"), 4, 0)
        self._spin_st_dep = QDoubleSpinBox()
        self._spin_st_dep.setRange(0.02, 0.5)
        self._spin_st_dep.setValue(0.1)
        self._spin_st_dep.setSingleStep(0.01)
        self._spin_st_dep.setDecimals(2)
        self._spin_st_dep.setFixedWidth(80)
        self._spin_st_dep.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_st_dep, 4, 1)
        left_grid.addWidget(_hint("mV(0.02~0.5)"), 4, 2)

        # Row 5 – ST Duration
        left_grid.addWidget(_lbl("ST Duration ≥"), 5, 0)
        self._spin_st_dur = QSpinBox()
        self._spin_st_dur.setRange(1, 300)
        self._spin_st_dur.setValue(60)
        self._spin_st_dur.setFixedWidth(80)
        self._spin_st_dur.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_st_dur, 5, 1)
        left_grid.addWidget(_hint("s(1~300)"), 5, 2)

        # Row 6 – ST Reset
        left_grid.addWidget(_lbl("ST Reset ≥"), 6, 0)
        self._spin_st_reset = QSpinBox()
        self._spin_st_reset.setRange(1, 300)
        self._spin_st_reset.setValue(60)
        self._spin_st_reset.setFixedWidth(80)
        self._spin_st_reset.setStyleSheet(spin_style)
        left_grid.addWidget(self._spin_st_reset, 6, 1)
        left_grid.addWidget(_hint("s(1~300)"), 6, 2)

        # Row 7 – bottom buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        for btn_lbl in ("Reanalysis", "Advanced", "Default"):
            btn = QPushButton(btn_lbl)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(30)
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        left_grid.addLayout(btn_row, 7, 0, 1, 3)
        left_grid.setRowStretch(8, 1)

        body_row.addWidget(left_widget)

        # ---- RIGHT: age-based Tachycardia / Bradycardia HR table ----
        right_widget = QWidget()
        right_widget.setStyleSheet("background: transparent;")
        right_grid = QGridLayout(right_widget)
        right_grid.setContentsMargins(0, 0, 0, 0)
        right_grid.setHorizontalSpacing(8)
        right_grid.setVerticalSpacing(10)

        # Column headers
        hdr_style = f"color:{UI_MUTED}; font-size:12px; font-weight:700;"
        h_age = QLabel("Age Group")
        h_age.setStyleSheet(hdr_style)
        h_tachy = QLabel("Tachycardia HR (bpm)")
        h_tachy.setStyleSheet(hdr_style)
        h_brady = QLabel("Bradycardia HR (bpm)")
        h_brady.setStyleSheet(hdr_style)
        right_grid.addWidget(h_age,   0, 0)
        right_grid.addWidget(h_tachy, 0, 1, 1, 3)
        right_grid.addWidget(h_brady, 0, 4, 1, 3)

        # Age rows: (label, tachy_default, brady_default, hint)
        age_rows = [
            ("≥16 years",         120, 50),
            ("11 years~15 years", 110, 60),
            ("6 years~10 years",  120, 70),
            ("4 years~5 years",   130, 80),
            ("2 years~3 years",   140, 80),
            ("2 months~1 year",   150, 90),
            ("7 days~1 month",    160, 90),
            ("≤6 days",           170, 110),
        ]

        self._spin_tachy_age = []
        self._spin_brady_age = []
        row_hint = f"bpm(20~250)"

        for r, (age_lbl, tachy_val, brady_val) in enumerate(age_rows, start=1):
            age_w = QLabel(age_lbl)
            age_w.setStyleSheet(f"color:{UI_TEXT}; font-size:12px;")
            right_grid.addWidget(age_w, r, 0)

            # ≥ symbol
            ge_t = QLabel("≥")
            ge_t.setStyleSheet(f"color:{UI_MUTED}; font-size:12px;")
            right_grid.addWidget(ge_t, r, 1)

            sp_tachy = QSpinBox()
            sp_tachy.setRange(20, 250)
            sp_tachy.setValue(tachy_val)
            sp_tachy.setFixedWidth(70)
            sp_tachy.setStyleSheet(spin_style)
            right_grid.addWidget(sp_tachy, r, 2)
            self._spin_tachy_age.append(sp_tachy)

            hint_t = QLabel(row_hint)
            hint_t.setStyleSheet(hint_style)
            right_grid.addWidget(hint_t, r, 3)

            # < symbol
            lt_b = QLabel("<")
            lt_b.setStyleSheet(f"color:{UI_MUTED}; font-size:12px;")
            right_grid.addWidget(lt_b, r, 4)

            sp_brady = QSpinBox()
            sp_brady.setRange(20, 250)
            sp_brady.setValue(brady_val)
            sp_brady.setFixedWidth(70)
            sp_brady.setStyleSheet(spin_style)
            right_grid.addWidget(sp_brady, r, 5)
            self._spin_brady_age.append(sp_brady)

            hint_b = QLabel(row_hint)
            hint_b.setStyleSheet(hint_style)
            right_grid.addWidget(hint_b, r, 6)

        right_grid.setRowStretch(len(age_rows) + 1, 1)
        body_row.addWidget(right_widget)

        layout.addLayout(body_row)
        layout.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _create_hrv_settings_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        spin_style = (
            f"background:{UI_PANEL_ALT}; color:{UI_TEXT}; "
            f"border:1px solid {UI_BORDER}; border-radius:4px; padding:2px 6px; font-size:12px;"
        )
        lbl_style  = f"color:{UI_TEXT}; font-size:12px;"
        hint_style = f"color:{UI_MUTED}; font-size:11px;"
        unit_style = f"color:{UI_MUTED}; font-size:12px;"

        # ──────────────────────────────────────────────
        # 1. HRV frequency domain analysis
        # ──────────────────────────────────────────────
        title_fd = QLabel("HRV frequency domain analysis")
        title_fd.setStyleSheet(
            f"font-size:13px; font-weight:700; color:{UI_ACCENT_HOVER}; "
            f"background:{UI_PANEL_ALT}; border:1px solid {UI_BORDER}; "
            f"border-radius:4px; padding:6px 10px;"
        )
        layout.addWidget(title_fd)

        fd_body = QHBoxLayout()
        fd_body.setSpacing(24)

        # Left: Analysis mode + Min/Max RRI
        fd_left = QWidget()
        fd_left.setStyleSheet("background: transparent;")
        fd_left_grid = QGridLayout(fd_left)
        fd_left_grid.setContentsMargins(0, 0, 0, 0)
        fd_left_grid.setHorizontalSpacing(10)
        fd_left_grid.setVerticalSpacing(12)

        # Analysis mode
        lbl_am = QLabel("Analysis mode")
        lbl_am.setStyleSheet(lbl_style)
        fd_left_grid.addWidget(lbl_am, 0, 0)
        self._combo_hrv_mode = QComboBox()
        self._combo_hrv_mode.addItems(["AR", "FFT"])
        self._combo_hrv_mode.setCurrentText("AR")
        self._combo_hrv_mode.setFixedWidth(90)
        self._combo_hrv_mode.setStyleSheet(f"""
            QComboBox {{
                background:{UI_PANEL_ALT}; color:{UI_TEXT};
                border:1px solid {UI_BORDER}; border-radius:4px; padding:3px 8px;
            }}
            QComboBox QAbstractItemView {{
                background:{UI_PANEL}; color:{UI_TEXT}; border:1px solid {UI_BORDER};
            }}
        """)
        fd_left_grid.addWidget(self._combo_hrv_mode, 0, 1)

        # Min RRI
        lbl_fd_min = QLabel("Min RRI\n(0~500ms)")
        lbl_fd_min.setStyleSheet(f"color:{UI_TEXT}; font-size:12px;")
        fd_left_grid.addWidget(lbl_fd_min, 1, 0)
        self._spin_fd_min_rri = QSpinBox()
        self._spin_fd_min_rri.setRange(0, 500)
        self._spin_fd_min_rri.setValue(300)
        self._spin_fd_min_rri.setFixedWidth(80)
        self._spin_fd_min_rri.setStyleSheet(spin_style)
        fd_left_grid.addWidget(self._spin_fd_min_rri, 1, 1)
        lbl_fd_min_u = QLabel("ms")
        lbl_fd_min_u.setStyleSheet(unit_style)
        fd_left_grid.addWidget(lbl_fd_min_u, 1, 2)

        # Max RRI
        lbl_fd_max = QLabel("Max RRI\n(500~5000ms)")
        lbl_fd_max.setStyleSheet(f"color:{UI_TEXT}; font-size:12px;")
        fd_left_grid.addWidget(lbl_fd_max, 2, 0)
        self._spin_fd_max_rri = QSpinBox()
        self._spin_fd_max_rri.setRange(500, 5000)
        self._spin_fd_max_rri.setValue(3000)
        self._spin_fd_max_rri.setFixedWidth(80)
        self._spin_fd_max_rri.setStyleSheet(spin_style)
        fd_left_grid.addWidget(self._spin_fd_max_rri, 2, 1)
        lbl_fd_max_u = QLabel("ms")
        lbl_fd_max_u.setStyleSheet(unit_style)
        fd_left_grid.addWidget(lbl_fd_max_u, 2, 2)

        fd_left_grid.setRowStretch(3, 1)
        fd_body.addWidget(fd_left)

        # Right: Power bands table (VLF / LF / HF)
        pb_frame = QFrame()
        pb_frame.setStyleSheet(
            f"QFrame {{ background:{UI_PANEL}; border:1px solid {UI_BORDER}; border-radius:6px; }}"
        )
        pb_grid = QGridLayout(pb_frame)
        pb_grid.setContentsMargins(14, 10, 14, 10)
        pb_grid.setHorizontalSpacing(10)
        pb_grid.setVerticalSpacing(10)

        # Power bands header
        hdr = QLabel("Power bands")
        hdr.setStyleSheet(f"font-weight:700; color:{UI_TEXT}; font-size:12px; border:none;")
        hdr.setAlignment(Qt.AlignCenter)
        pb_grid.addWidget(hdr, 0, 0, 1, 5)

        for col, txt in enumerate(["", "Start", "", "Stop", ""]):
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{UI_MUTED}; font-size:11px; font-weight:700; border:none;")
            lbl.setAlignment(Qt.AlignCenter)
            pb_grid.addWidget(lbl, 1, col)

        # VLF / LF / HF rows: (name, start_default, stop_default, range_hint)
        pb_rows = [
            ("VLF:", 0.003, 0.04,  "(0.0~0.5)"),
            ("LF:",  0.04,  0.15,  "(0.0~0.5)"),
            ("HF:",  0.15,  0.4,   "(0.0~0.5)"),
        ]
        self._spin_pb_start = []
        self._spin_pb_stop  = []

        for r, (band, start_val, stop_val, rng) in enumerate(pb_rows, start=2):
            lbl_band = QLabel(band)
            lbl_band.setStyleSheet(f"color:{UI_TEXT}; font-size:12px; font-weight:700; border:none;")
            pb_grid.addWidget(lbl_band, r, 0)

            sp_start = QDoubleSpinBox()
            sp_start.setRange(0.0, 0.5)
            sp_start.setValue(start_val)
            sp_start.setDecimals(3)
            sp_start.setSingleStep(0.001)
            sp_start.setFixedWidth(75)
            sp_start.setStyleSheet(spin_style)
            pb_grid.addWidget(sp_start, r, 1)
            self._spin_pb_start.append(sp_start)

            tilde = QLabel("~")
            tilde.setStyleSheet(f"color:{UI_MUTED}; border:none;")
            tilde.setAlignment(Qt.AlignCenter)
            pb_grid.addWidget(tilde, r, 2)

            sp_stop = QDoubleSpinBox()
            sp_stop.setRange(0.0, 0.5)
            sp_stop.setValue(stop_val)
            sp_stop.setDecimals(3)
            sp_stop.setSingleStep(0.001)
            sp_stop.setFixedWidth(75)
            sp_stop.setStyleSheet(spin_style)
            pb_grid.addWidget(sp_stop, r, 3)
            self._spin_pb_stop.append(sp_stop)

            hz_lbl = QLabel("Hz")
            hz_lbl.setStyleSheet(f"color:{UI_MUTED}; font-size:12px; border:none;")
            pb_grid.addWidget(hz_lbl, r, 4)

        # Range hints row
        for col_idx, rng in enumerate(["", "(0.0~0.5)", "", "(0.0~0.5)", ""]):
            rng_lbl = QLabel(rng)
            rng_lbl.setStyleSheet(f"color:{UI_MUTED}; font-size:10px; border:none;")
            rng_lbl.setAlignment(Qt.AlignCenter)
            pb_grid.addWidget(rng_lbl, 5, col_idx)

        fd_body.addWidget(pb_frame)
        fd_body.addStretch(1)
        layout.addLayout(fd_body)

        # ──────────────────────────────────────────────
        # 2. HRV time domain analysis
        # ──────────────────────────────────────────────
        title_td = QLabel("HRV time domain analysis")
        title_td.setStyleSheet(
            f"font-size:13px; font-weight:700; color:{UI_ACCENT_HOVER}; "
            f"background:{UI_PANEL_ALT}; border:1px solid {UI_BORDER}; "
            f"border-radius:4px; padding:6px 10px;"
        )
        layout.addWidget(title_td)

        td_body = QHBoxLayout()
        td_body.setSpacing(32)

        # Left: Min/Max RRI (time domain)
        td_left = QWidget()
        td_left.setStyleSheet("background: transparent;")
        td_left_grid = QGridLayout(td_left)
        td_left_grid.setContentsMargins(0, 0, 0, 0)
        td_left_grid.setHorizontalSpacing(10)
        td_left_grid.setVerticalSpacing(12)

        lbl_td_min = QLabel("Min RRI\n(0~500ms)")
        lbl_td_min.setStyleSheet(lbl_style)
        td_left_grid.addWidget(lbl_td_min, 0, 0)
        self._spin_td_min_rri = QSpinBox()
        self._spin_td_min_rri.setRange(0, 500)
        self._spin_td_min_rri.setValue(300)
        self._spin_td_min_rri.setFixedWidth(80)
        self._spin_td_min_rri.setStyleSheet(spin_style)
        td_left_grid.addWidget(self._spin_td_min_rri, 0, 1)
        lbl_td_min_u = QLabel("ms")
        lbl_td_min_u.setStyleSheet(unit_style)
        td_left_grid.addWidget(lbl_td_min_u, 0, 2)

        lbl_td_max = QLabel("Max RRI\n(500~5000ms)")
        lbl_td_max.setStyleSheet(lbl_style)
        td_left_grid.addWidget(lbl_td_max, 1, 0)
        self._spin_td_max_rri = QSpinBox()
        self._spin_td_max_rri.setRange(500, 5000)
        self._spin_td_max_rri.setValue(3000)
        self._spin_td_max_rri.setFixedWidth(80)
        self._spin_td_max_rri.setStyleSheet(spin_style)
        td_left_grid.addWidget(self._spin_td_max_rri, 1, 1)
        lbl_td_max_u = QLabel("ms")
        lbl_td_max_u.setStyleSheet(unit_style)
        td_left_grid.addWidget(lbl_td_max_u, 1, 2)

        td_left_grid.setRowStretch(2, 1)
        td_body.addWidget(td_left)

        # Right: Day / Night hour ranges
        dn_widget = QWidget()
        dn_widget.setStyleSheet("background: transparent;")
        dn_grid = QGridLayout(dn_widget)
        dn_grid.setContentsMargins(0, 0, 0, 0)
        dn_grid.setHorizontalSpacing(8)
        dn_grid.setVerticalSpacing(12)

        # Day row
        lbl_day = QLabel("Day:")
        lbl_day.setStyleSheet(lbl_style)
        dn_grid.addWidget(lbl_day, 0, 0)

        self._spin_day_start = QSpinBox()
        self._spin_day_start.setRange(0, 23)
        self._spin_day_start.setValue(13)
        self._spin_day_start.setFixedWidth(60)
        self._spin_day_start.setStyleSheet(spin_style)
        dn_grid.addWidget(self._spin_day_start, 0, 1)

        tilde_d = QLabel("~")
        tilde_d.setStyleSheet(unit_style)
        dn_grid.addWidget(tilde_d, 0, 2)

        self._spin_day_end = QSpinBox()
        self._spin_day_end.setRange(0, 23)
        self._spin_day_end.setValue(17)
        self._spin_day_end.setFixedWidth(60)
        self._spin_day_end.setStyleSheet(spin_style)
        dn_grid.addWidget(self._spin_day_end, 0, 3)

        h_day = QLabel("H")
        h_day.setStyleSheet(f"color:{UI_ACCENT}; font-size:13px; font-weight:700;")
        dn_grid.addWidget(h_day, 0, 4)

        # Night row
        lbl_night = QLabel("Night:")
        lbl_night.setStyleSheet(lbl_style)
        dn_grid.addWidget(lbl_night, 1, 0)

        self._spin_night_start = QSpinBox()
        self._spin_night_start.setRange(0, 23)
        self._spin_night_start.setValue(23)
        self._spin_night_start.setFixedWidth(60)
        self._spin_night_start.setStyleSheet(spin_style)
        dn_grid.addWidget(self._spin_night_start, 1, 1)

        tilde_n = QLabel("~")
        tilde_n.setStyleSheet(unit_style)
        dn_grid.addWidget(tilde_n, 1, 2)

        self._spin_night_end = QSpinBox()
        self._spin_night_end.setRange(0, 23)
        self._spin_night_end.setValue(5)
        self._spin_night_end.setFixedWidth(60)
        self._spin_night_end.setStyleSheet(spin_style)
        dn_grid.addWidget(self._spin_night_end, 1, 3)

        h_night = QLabel("H")
        h_night.setStyleSheet(f"color:{UI_ACCENT}; font-size:13px; font-weight:700;")
        dn_grid.addWidget(h_night, 1, 4)

        dn_grid.setRowStretch(2, 1)
        td_body.addWidget(dn_widget)
        td_body.addStretch(1)
        layout.addLayout(td_body)

        # Default button (bottom right, matching reference)
        btn_default_row = QHBoxLayout()
        btn_default_row.addStretch(1)
        btn_hrv_default = QPushButton("Default")
        btn_hrv_default.setStyleSheet(_style_btn())
        btn_hrv_default.setFixedWidth(100)
        btn_hrv_default.setFixedHeight(30)
        btn_hrv_default.clicked.connect(self._on_hrv_default_clicked)
        btn_default_row.addWidget(btn_hrv_default)
        layout.addLayout(btn_default_row)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _on_hrv_default_clicked(self):
        """Reset HRV settings to defaults."""
        self._combo_hrv_mode.setCurrentText("AR")
        self._spin_fd_min_rri.setValue(300)
        self._spin_fd_max_rri.setValue(3000)
        defaults_start = [0.003, 0.04, 0.15]
        defaults_stop  = [0.04,  0.15, 0.4]
        for i, (sp_s, sp_e) in enumerate(zip(self._spin_pb_start, self._spin_pb_stop)):
            sp_s.setValue(defaults_start[i])
            sp_e.setValue(defaults_stop[i])
        self._spin_td_min_rri.setValue(300)
        self._spin_td_max_rri.setValue(3000)
        self._spin_day_start.setValue(13)
        self._spin_day_end.setValue(17)
        self._spin_night_start.setValue(23)
        self._spin_night_end.setValue(5)


    def get_settings(self) -> Dict[str, Any]:
        lead_gains = {}
        lead_inverts = {}

        # Check if "All" row has a gain selected
        all_gain = None
        if "All" in self._gain_radios:
            for idx, val in enumerate(GAIN_VALUES):
                if self._gain_radios["All"][idx].isChecked():
                    all_gain = float(val)
                    break

        # Check if "All" row invert is checked
        all_invert = self._invert_checks.get("All", None) and self._invert_checks["All"].isChecked()

        for lead in LEADS:
            if all_gain is not None:
                lead_gains[lead] = all_gain
            else:
                for idx, val in enumerate(GAIN_VALUES):
                    if self._gain_radios[lead][idx].isChecked():
                        lead_gains[lead] = float(val)

            if all_invert:
                lead_inverts[lead] = True
            else:
                lead_inverts[lead] = self._invert_checks[lead].isChecked()

        age_labels = [
            ">= 16 years", "11-15 years", "6-10 years", "4-5 years",
            "2-3 years", "2mo-1yr", "7d-1mo", "<=6 days"
        ]
        tachy_by_age = {age_labels[i]: sp.value() for i, sp in enumerate(self._spin_tachy_age)}
        brady_by_age = {age_labels[i]: sp.value() for i, sp in enumerate(self._spin_brady_age)}

        return {
            "lead_gains": lead_gains,
            "lead_inverts": lead_inverts,
            "strip_width_sec": int(self._combo_width.currentText()),
            "pause_sec": self._spin_pause.value(),
            "atrial_prematurity_pct": self._spin_atrial_prem.value(),
            "atrial_tach_hr_bpm": self._spin_atrial_tach_hr.value(),
            "st_elevation_mv": self._spin_st_elev.value(),
            "st_depression_mv": self._spin_st_dep.value(),
            "st_duration_sec": self._spin_st_dur.value(),
            "st_reset_sec": self._spin_st_reset.value(),
            "tachy_hr_by_age": tachy_by_age,
            "brady_hr_by_age": brady_by_age,
            "hrv_mode": self._combo_hrv_mode.currentText(),
            "hrv_fd_min_rri_ms": self._spin_fd_min_rri.value(),
            "hrv_fd_max_rri_ms": self._spin_fd_max_rri.value(),
            "hrv_vlf_start": self._spin_pb_start[0].value(),
            "hrv_vlf_stop":  self._spin_pb_stop[0].value(),
            "hrv_lf_start":  self._spin_pb_start[1].value(),
            "hrv_lf_stop":   self._spin_pb_stop[1].value(),
            "hrv_hf_start":  self._spin_pb_start[2].value(),
            "hrv_hf_stop":   self._spin_pb_stop[2].value(),
            "hrv_td_min_rri_ms": self._spin_td_min_rri.value(),
            "hrv_td_max_rri_ms": self._spin_td_max_rri.value(),
            "hrv_day_start_h":   self._spin_day_start.value(),
            "hrv_day_end_h":     self._spin_day_end.value(),
            "hrv_night_start_h": self._spin_night_start.value(),
            "hrv_night_end_h":   self._spin_night_end.value(),
        }

    def _on_reset_clicked(self):
        for lead in LEADS:
            # Default to 10 mm/mV (index 1 in GAIN_VALUES)
            self._gain_radios[lead][1].setChecked(True)
            self._invert_checks[lead].setChecked(False)

        self._combo_width.setCurrentText("14")

        # Reset arrhythmia params
        self._spin_pause.setValue(2.0)
        self._spin_atrial_prem.setValue(20)
        self._spin_atrial_tach_hr.setValue(90)
        self._spin_st_elev.setValue(0.3)
        self._spin_st_dep.setValue(0.1)
        self._spin_st_dur.setValue(60)
        self._spin_st_reset.setValue(60)

        # Reset age-based tachy/brady defaults
        tachy_defaults = [120, 110, 120, 130, 140, 150, 160, 170]
        brady_defaults = [50, 60, 70, 80, 80, 90, 90, 110]
        for i, sp in enumerate(self._spin_tachy_age):
            sp.setValue(tachy_defaults[i])
        for i, sp in enumerate(self._spin_brady_age):
            sp.setValue(brady_defaults[i])

        # Reset HRV params – delegate to the dedicated handler
        self._on_hrv_default_clicked()

        QMessageBox.information(self, "Record Settings", "Settings have been reset to default values.")

    def _on_apply_clicked(self):
        settings = self.get_settings()
        self.settings_changed.emit(settings)
        QMessageBox.information(self, "Record Settings", "Record settings successfully applied.")

    def _on_ok_clicked(self):
        self._on_apply_clicked()

    def _on_cancel_clicked(self):
        pass
