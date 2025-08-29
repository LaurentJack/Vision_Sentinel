import sys, os, time, yaml, csv, json
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from .detection import YOLODetector
from .tracker import SimpleTracker
from .face_recog import FaceRecognizer
from .utils import put_fps, draw_polygon, point_in_polygon, center_of, draw_line, segment_intersection, side_of_line, infer_color_for_box

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
SOURCES_PATH = os.path.join(os.path.dirname(__file__), "sources.json")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

class VideoWidget(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #000;")
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        p = ev.position().toPoint()
        self.clicked.emit(p.x(), p.y())

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Sentinel by Laurent Jacquemyns")
        self.resize(1450, 880)
        self.cfg = load_config()
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.frame = None
        self.frame_idx = 0

        wcfg = self.cfg.get("weapons", {}) or {}
        self.detector = YOLODetector(
            model_name=self.cfg["yolo"]["model"],
            classes_of_interest=self.cfg["yolo"]["classes_of_interest"],
            conf=self.cfg["yolo"]["conf_threshold"],
            iou=self.cfg["yolo"]["iou_threshold"],
            weapons_enabled=wcfg.get("enabled", False),
            weapons_model=wcfg.get("model"),
            weapons_conf=wcfg.get("conf_threshold", 0.5),
            weapons_iou=wcfg.get("iou_threshold", 0.5),
            weapons_aliases=wcfg.get("class_aliases", {})
        )
        self.tracker = SimpleTracker(max_age_frames=30, iou_threshold=0.35)
        self.facerec = FaceRecognizer(
            model_name=self.cfg["face"]["model_name"],
            provider=self.cfg["face"]["provider"],
            threshold=float(self.cfg["face"].get("similarity_threshold", 0.35)),
            facebank_dir=os.path.join(os.path.dirname(__file__), "data", "facebank")
        )

        self.detections: List[Dict[str, Any]] = []
        self.tracked: List[Dict[str, Any]] = []
        self.prev_centers: Dict[int, Tuple[int,int]] = {}

        # Zones
        self.zones: List[Dict[str, Any]] = self.cfg.get("zones", [])
        if not isinstance(self.zones, list): self.zones = []
        self.active_zone_idx = 0 if self.zones else -1
        self.zone_inside_ids: List[set] = [set() for _ in self.zones]
        self.zone_unique_ids: List[set] = [set() for _ in self.zones]

        # Lines
        self.lines: List[Dict[str, Any]] = self.cfg.get("lines", [])
        if not isinstance(self.lines, list): self.lines = []
        for ln in self.lines:
            ln.setdefault("counts", {"AtoB":0, "BtoA":0})
        self.line_editing = False
        self.line_click_stage = 0
        self.pending_line = {"name":"", "p1":None, "p2":None, "counts":{"AtoB":0,"BtoA":0}}

        self.log_path = os.path.join(os.path.dirname(__file__), "..", self.cfg["logging"]["csv_path"])
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.ensure_csv()
    
        # UI
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: video
        left = QtWidgets.QVBoxLayout()
        self.video = VideoWidget()
        left.addWidget(self.video, 1)
        layout.addLayout(left, 3)

        # Right: controls
        right = QtWidgets.QVBoxLayout()

        # Source vidéo (avec flux prédéfinis)
        src_group = QtWidgets.QGroupBox("Source vidéo")
        src_layout = QtWidgets.QFormLayout(src_group)

        # 1) Liste déroulante des flux prédéfinis
        self.cmb_streams = QtWidgets.QComboBox()
        src_layout.addRow("Flux prédéfinis :", self.cmb_streams)
        

        # 2) Contrôles "custom" existants (cam index / URL)
        self.cmb_cam = QtWidgets.QComboBox(); self.cmb_cam.addItems([str(i) for i in range(0, 8)])
        self.txt_url = QtWidgets.QLineEdit(); self.txt_url.setPlaceholderText("rtsp://... ou fichier.mp4")
        src_layout.addRow("Caméra index :", self.cmb_cam)
        src_layout.addRow("URL :", self.txt_url)

        # 3) Boutons Start/Stop
        self.btn_start = QtWidgets.QPushButton("Démarrer")
        self.btn_stop  = QtWidgets.QPushButton("Arrêter")
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_stop)
        src_layout.addRow(hb)

        # 4) Gestion des flux (Ajouter / Modifier / Supprimer / Recharger)
        self.btn_stream_add    = QtWidgets.QPushButton("Ajouter flux…")
        self.btn_stream_edit   = QtWidgets.QPushButton("Modifier…")
        self.btn_stream_del    = QtWidgets.QPushButton("Supprimer")
        self.btn_stream_reload = QtWidgets.QPushButton("Recharger")
        hb2 = QtWidgets.QHBoxLayout()
        hb2.addWidget(self.btn_stream_add); hb2.addWidget(self.btn_stream_edit)
        hb2.addWidget(self.btn_stream_del); hb2.addWidget(self.btn_stream_reload)
        src_layout.addRow(hb2)

        right.addWidget(src_group)
        
        # Flux prédéfinis
        self.sources = self.load_sources()
        self.refresh_streams_combo()
        self.cmb_streams.currentIndexChanged.connect(self.on_stream_changed)
        self.btn_stream_add.clicked.connect(self.stream_add)
        self.btn_stream_edit.clicked.connect(self.stream_edit)
        self.btn_stream_del.clicked.connect(self.stream_del)
        self.btn_stream_reload.clicked.connect(self.stream_reload)
        
        # Zones group
        zone_group = QtWidgets.QGroupBox("Zones (polygones)")
        zlay = QtWidgets.QGridLayout(zone_group)
        self.chk_edit_zone = QtWidgets.QCheckBox("Mode Édition")
        self.cmb_zone = QtWidgets.QComboBox()
        self.btn_zone_new = QtWidgets.QPushButton("Nouvelle zone")
        self.btn_zone_del = QtWidgets.QPushButton("Supprimer zone")
        self.btn_zone_clear = QtWidgets.QPushButton("Effacer points")
        self.btn_save = QtWidgets.QPushButton("Sauver config")
        zlay.addWidget(self.chk_edit_zone, 0, 0, 1, 2)
        zlay.addWidget(QtWidgets.QLabel("Zone active:"), 1, 0); zlay.addWidget(self.cmb_zone, 1, 1)
        zlay.addWidget(self.btn_zone_new, 2, 0); zlay.addWidget(self.btn_zone_del, 2, 1)
        zlay.addWidget(self.btn_zone_clear, 3, 0); zlay.addWidget(self.btn_save, 3, 1)
        right.addWidget(zone_group)

        # Lines group
        line_group = QtWidgets.QGroupBox("Lignes virtuelles (A→B)")
        llay = QtWidgets.QGridLayout(line_group)
        self.lst_lines = QtWidgets.QListWidget()
        self.btn_line_new = QtWidgets.QPushButton("Nouvelle ligne")
        self.btn_line_del = QtWidgets.QPushButton("Supprimer ligne")
        self.lbl_line_hint = QtWidgets.QLabel("Cliquez 2 points sur la vidéo pour définir A puis B.")
        llay.addWidget(self.lst_lines, 0, 0, 3, 2)
        llay.addWidget(self.btn_line_new, 3, 0); llay.addWidget(self.btn_line_del, 3, 1)
        llay.addWidget(self.lbl_line_hint, 4, 0, 1, 2)
        right.addWidget(line_group, 1)

        # Face group
        face_group = QtWidgets.QGroupBox("Reconnaissance faciale")
        fgl = QtWidgets.QFormLayout(face_group)
        self.txt_name = QtWidgets.QLineEdit()
        self.btn_enroll = QtWidgets.QPushButton("Capturer & Enregistrer")
        self.btn_enroll_burst = QtWidgets.QPushButton("Capturer x10")
        self.btn_reload_gallery = QtWidgets.QPushButton("Recharger galerie")
        self.chk_blur_unknown = QtWidgets.QCheckBox("Flouter inconnus")
        self.chk_blur_unknown.setChecked(bool(self.cfg["drawing"]["blur_unknown_faces"]))
        self.spin_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_thresh.setRange(0.20, 0.80); self.spin_thresh.setSingleStep(0.01); self.spin_thresh.setDecimals(2)
        self.spin_thresh.setValue(float(self.cfg.get("face",{}).get("similarity_threshold", 0.35)))
        fgl.addRow("Nom:", self.txt_name)
        fgl.addRow(self.btn_enroll); fgl.addRow(self.btn_enroll_burst)
        fgl.addRow("Seuil simil.", self.spin_thresh)
        fgl.addRow(self.chk_blur_unknown)
        fgl.addRow(self.btn_reload_gallery)
        right.addWidget(face_group)

        # Options
        opt_group = QtWidgets.QGroupBox("Affichage")
        og = QtWidgets.QVBoxLayout(opt_group)
        self.chk_show_objects = QtWidgets.QCheckBox("Afficher objets/vêtements")
        self.chk_show_objects.setChecked(bool(self.cfg["drawing"].get("show_objects", True)))
        og.addWidget(self.chk_show_objects)
        right.addWidget(opt_group)

        # Stats
        stats_group = QtWidgets.QGroupBox("Statistiques")
        slay = QtWidgets.QVBoxLayout(stats_group)
        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.tbl_zone = QtWidgets.QTableWidget(0, 3); self.tbl_zone.setHorizontalHeaderLabels(["Zone", "Dans la zone", "Entrées uniques"])
        self.tbl_zone.horizontalHeader().setStretchLastSection(True); self.tbl_zone.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_lines = QtWidgets.QTableWidget(0, 3); self.tbl_lines.setHorizontalHeaderLabels(["Ligne (A→B)", "A→B", "B→A"])
        self.tbl_lines.horizontalHeader().setStretchLastSection(True); self.tbl_lines.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        slay.addWidget(self.lbl_fps); slay.addWidget(QtWidgets.QLabel("Zones:")); slay.addWidget(self.tbl_zone)
        slay.addWidget(QtWidgets.QLabel("Lignes:")); slay.addWidget(self.tbl_lines)
        right.addWidget(stats_group, 2)

        # Footer
        self.btn_quit = QtWidgets.QPushButton("Quitter")
        right.addWidget(self.btn_quit)

        layout.addLayout(right, 2)

        # Signals
        self.video.clicked.connect(self.on_video_click)
        self.btn_start.clicked.connect(self.start_capture); self.btn_stop.clicked.connect(self.stop_capture); self.btn_quit.clicked.connect(self.close)
        self.btn_zone_new.clicked.connect(self.zone_new); self.btn_zone_del.clicked.connect(self.zone_del); self.btn_zone_clear.clicked.connect(self.zone_clear)
        self.btn_save.clicked.connect(self.save_all); self.cmb_zone.currentIndexChanged.connect(self.on_zone_changed)
        self.btn_line_new.clicked.connect(self.line_new); self.btn_line_del.clicked.connect(self.line_del)
        self.btn_enroll.clicked.connect(self.enroll_face); self.btn_enroll_burst.clicked.connect(self.enroll_burst); self.btn_reload_gallery.clicked.connect(self.reload_gallery)
        self.spin_thresh.valueChanged.connect(self.on_threshold_changed)

        self.refresh_zone_combo(); self.refresh_line_list(); self.refresh_tables()

    def ensure_csv(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","event","track_id","zone_or_line","direction"])
    
    # --- Gestion des flux prédéfinis (sources.json) ---
    def load_sources(self):
        """Charge la liste depuis sources.json, ou crée des valeurs par défaut."""
        try:
            if not os.path.exists(SOURCES_PATH):
                defaults = [
                    {"name": "Webcam 0", "source": "0"},
                    {"name": "Démo RTSP", "source": "rtsp://192.168.1.50:554/stream1"}
                ]
                with open(SOURCES_PATH, "w", encoding="utf-8") as f:
                    json.dump(defaults, f, ensure_ascii=False, indent=2)
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("sources.json mal formé (liste attendue)")
                # normaliser
                out = []
                for s in data:
                    name = (s.get("name","") or "").strip()
                    src  = (s.get("source","") or "").strip()
                    if name and src:
                        out.append({"name": name, "source": src})
                return out
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Flux prédéfinis",
                                          f"Erreur de lecture de {SOURCES_PATH} : {e}")
            return []

    def save_sources(self):
        """Écrit self.sources dans sources.json."""
        try:
            with open(SOURCES_PATH, "w", encoding="utf-8") as f:
                json.dump(self.sources, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Flux prédéfinis",
                                           f"Impossible d'écrire {SOURCES_PATH} : {e}")

    def refresh_streams_combo(self):
        """Recharge la combo à partir de self.sources."""
        self.cmb_streams.blockSignals(True)
        self.cmb_streams.clear()
        for s in self.sources:
            self.cmb_streams.addItem(s.get("name", "(sans nom)"))
        self.cmb_streams.blockSignals(False)
        if self.sources:
            self.cmb_streams.setCurrentIndex(0)
            self.on_stream_changed(0)

    def get_selected_source(self):
        """Retourne la source du flux sélectionné (index caméra ou URL)."""
        idx = self.cmb_streams.currentIndex()
        if 0 <= idx < len(self.sources):
            return (self.sources[idx].get("source", "") or "").strip()
        return ""

    def on_stream_changed(self, idx):
        """Quand on change de flux dans la combo, refléter sur cam index / URL."""
        src = self.get_selected_source()
        if src.isdigit():
            self.cmb_cam.setCurrentText(src)   # index caméra
            self.txt_url.setText("")
        else:
            self.txt_url.setText(src)          # URL/fichier

    def stream_add(self):
        """Ajouter un flux via deux boîtes de dialogue."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Ajouter un flux", "Nom du flux :")
        if not ok or not name.strip():
            return
        src, ok = QtWidgets.QInputDialog.getText(self, "Ajouter un flux",
                                                 "Source (index caméra ou URL) :")
        if not ok or not src.strip():
            return
        self.sources.append({"name": name.strip(), "source": src.strip()})
        self.save_sources()
        self.refresh_streams_combo()
        self.cmb_streams.setCurrentIndex(len(self.sources) - 1)
        self.on_stream_changed(len(self.sources) - 1)
        QtWidgets.QMessageBox.information(self, "Flux", f"Ajouté : {name.strip()}")

    def stream_edit(self):
        """Modifier le flux sélectionné."""
        idx = self.cmb_streams.currentIndex()
        if idx < 0 or idx >= len(self.sources):
            return
        cur = self.sources[idx]
        name, ok = QtWidgets.QInputDialog.getText(self, "Modifier le flux", "Nom du flux :",
                                                  text=cur.get("name", ""))
        if not ok or not name.strip():
            return
        src, ok = QtWidgets.QInputDialog.getText(self, "Modifier le flux",
                                                 "Source (index caméra ou URL) :",
                                                 text=cur.get("source", ""))
        if not ok or not src.strip():
            return
        self.sources[idx] = {"name": name.strip(), "source": src.strip()}
        self.save_sources()
        self.refresh_streams_combo()
        self.cmb_streams.setCurrentIndex(idx)
        self.on_stream_changed(idx)
        QtWidgets.QMessageBox.information(self, "Flux", f"Modifié : {name.strip()}")

    def stream_del(self):
        """Supprimer le flux sélectionné."""
        idx = self.cmb_streams.currentIndex()
        if idx < 0 or idx >= len(self.sources):
            return
        rep = QtWidgets.QMessageBox.question(
            self, "Supprimer",
            f"Supprimer « {self.sources[idx].get('name', '(sans nom)')} » ?"
        )
        if rep != QtWidgets.QMessageBox.Yes:
            return
        del self.sources[idx]
        self.save_sources()
        self.refresh_streams_combo()

    def stream_reload(self):
        """Relire sources.json depuis le disque."""
        self.sources = self.load_sources()
        self.refresh_streams_combo()
    # --- fin gestion des flux ---

    
    # Face handlers
    def on_threshold_changed(self, v):
        self.facerec.set_threshold(float(v))
        self.cfg["face"]["similarity_threshold"] = float(v)

    def reload_gallery(self):
        try:
            self.facerec.load_facebank()
            QtWidgets.QMessageBox.information(self, "Galerie", "Galerie rechargée.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    def enroll_burst(self):
        if self.cap is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Source vidéo arrêtée.")
            return
        name = self.txt_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez saisir un nom.")
            return
        saved = 0
        for _ in range(10):
            ok, f = self.cap.read()
            if not ok or f is None:
                continue
            try:
                if self.facerec.enroll_from_frame(f, name=name, face_index=0):
                    saved += 1
            except:
                pass
        QtWidgets.QMessageBox.information(self, "Enrôlement x10", f"{saved} images enregistrées pour {name}.")

    def enroll_face(self):
        if self.frame is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Pas d'image vidéo.")
            return
        name = self.txt_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez saisir un nom.")
            return
        ok = False
        try:
            ok = self.facerec.enroll_from_frame(self.frame, name=name, face_index=0)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur d'enrôlement", str(e))
            return
        QtWidgets.QMessageBox.information(self, "Enrôlement", "Succès." if ok else "Aucun visage détecté.")

    # UI helpers
    def refresh_zone_combo(self):
        self.cmb_zone.blockSignals(True); self.cmb_zone.clear()
        for z in self.zones: self.cmb_zone.addItem(z.get("name","Zone"))
        self.cmb_zone.blockSignals(False)
        if self.zones:
            if self.active_zone_idx < 0: self.active_zone_idx = 0
            self.cmb_zone.setCurrentIndex(self.active_zone_idx)

    def refresh_line_list(self):
        self.lst_lines.clear()
        for ln in self.lines:
            counts = ln.get("counts", {"AtoB":0,"BtoA":0})
            self.lst_lines.addItem(f'{ln.get("name","Ligne")} — A→B:{counts["AtoB"]} | B→A:{counts["BtoA"]}')

    def refresh_tables(self):
        self.tbl_zone.setRowCount(len(self.zones))
        for i, z in enumerate(self.zones):
            inside = len(self.zone_inside_ids[i]) if i < len(self.zone_inside_ids) else 0
            unique = len(self.zone_unique_ids[i]) if i < len(self.zone_unique_ids) else 0
            self.tbl_zone.setItem(i, 0, QtWidgets.QTableWidgetItem(z.get("name","Zone")))
            self.tbl_zone.setItem(i, 1, QtWidgets.QTableWidgetItem(str(inside)))
            self.tbl_zone.setItem(i, 2, QtWidgets.QTableWidgetItem(str(unique)))
        self.tbl_lines.setRowCount(len(self.lines))
        for i, ln in enumerate(self.lines):
            counts = ln.get("counts", {"AtoB":0,"BtoA":0})
            self.tbl_lines.setItem(i, 0, QtWidgets.QTableWidgetItem(ln.get("name","Ligne")))
            self.tbl_lines.setItem(i, 1, QtWidgets.QTableWidgetItem(str(counts.get("AtoB",0))))
            self.tbl_lines.setItem(i, 2, QtWidgets.QTableWidgetItem(str(counts.get("BtoA",0))))

    # Zones
    def zone_new(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Nouvelle zone", "Nom de la zone:", text=f"Zone {len(self.zones)+1}")
        if not ok: return
        self.zones.append({"name": name, "polygon": []})
        self.zone_inside_ids.append(set()); self.zone_unique_ids.append(set())
        self.active_zone_idx = len(self.zones)-1
        self.refresh_zone_combo(); self.refresh_tables()

    def zone_del(self):
        idx = self.cmb_zone.currentIndex()
        if idx < 0: return
        del self.zones[idx]; del self.zone_inside_ids[idx]; del self.zone_unique_ids[idx]
        self.active_zone_idx = min(idx, len(self.zones)-1)
        self.refresh_zone_combo(); self.refresh_tables()

    def zone_clear(self):
        idx = self.cmb_zone.currentIndex()
        if idx < 0: return
        self.zones[idx]["polygon"] = []
        self.refresh_tables()

    def on_zone_changed(self, idx):
        self.active_zone_idx = idx

    # Lines
    def line_new(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Nouvelle ligne", "Nom de la ligne:", text=f"Ligne {len(self.lines)+1}")
        if not ok: return
        self.pending_line = {"name": name, "p1": None, "p2": None, "counts":{"AtoB":0,"BtoA":0}}
        self.line_editing = True; self.line_click_stage = 1
        QtWidgets.QMessageBox.information(self, "Ligne virtuelle", "Cliquez le point A puis le point B sur la vidéo.")

    def line_del(self):
        row = self.lst_lines.currentRow()
        if row < 0: return
        del self.lines[row]
        self.refresh_line_list(); self.refresh_tables()

    # Video click (zones + lines)
    def on_video_click(self, x, y):
        if self.frame is None: return
        pix = self.video.pixmap()
        if pix is None: return
        lbl_w, lbl_h = self.video.width(), self.video.height()
        img_h, img_w = self.frame.shape[:2]
        scale = min(lbl_w / img_w, lbl_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        off_x = (lbl_w - new_w) // 2; off_y = (lbl_h - new_h) // 2
        if not (off_x <= x <= off_x+new_w and off_y <= y <= off_y+new_h): return
        fx = int((x - off_x) / scale); fy = int((y - off_y) / scale)

        if self.line_editing:
            if self.line_click_stage == 1:
                self.pending_line["p1"] = (fx, fy); self.line_click_stage = 2; return
            elif self.line_click_stage == 2:
                self.pending_line["p2"] = (fx, fy)
                if self.pending_line["p1"] and self.pending_line["p2"]:
                    self.lines.append(self.pending_line)
                    self.line_editing = False; self.line_click_stage = 0
                    self.pending_line = {"name":"", "p1":None, "p2":None, "counts":{"AtoB":0,"BtoA":0}}
                    self.refresh_line_list(); self.refresh_tables()
                return

        if self.chk_edit_zone.isChecked() and 0 <= self.active_zone_idx < len(self.zones):
            self.zones[self.active_zone_idx]["polygon"].append((fx, fy))
            self.refresh_tables()

    def start_capture(self):
        # Priorité : flux prédéfini (combo) > URL saisie > index caméra
        chosen = self.get_selected_source().strip()
        if chosen:
            if chosen.isdigit():
                self.cap = cv2.VideoCapture(int(chosen))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg["video"]["width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["video"]["height"])
            else:
                self.cap = cv2.VideoCapture(chosen)
        else:
            url = self.txt_url.text().strip()
            if url:
                self.cap = cv2.VideoCapture(url)
            else:
                self.cap = cv2.VideoCapture(int(self.cmb_cam.currentText()))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg["video"]["width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["video"]["height"])

        if not self.cap or not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Erreur", "Impossible d'ouvrir la source vidéo.")
            self.cap = None
            return
        self.timer.start(1)



    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release(); self.cap = None

    def log_event(self, event: str, track_id: int, ref: str, direction: str = ""):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([int(time.time()), event, track_id, ref, direction])

    def save_all(self):
        """Persist UI settings and geometry to config.yaml."""
        try:
            # Zones polygons
            self.cfg["zones"] = [
                {"name": z.get("name","Zone"),
                 "polygon": [[int(x), int(y)] for (x,y) in z.get("polygon", [])]}
                for z in self.zones
            ]
            # Lines endpoints
            self.cfg["lines"] = [
                {"name": ln.get("name","Ligne"),
                 "p1": list(ln.get("p1", [])) if ln.get("p1") else [],
                 "p2": list(ln.get("p2", [])) if ln.get("p2") else []}
                for ln in self.lines
            ]
            # Drawing & face options
            self.cfg["drawing"]["blur_unknown_faces"] = bool(self.chk_blur_unknown.isChecked())
            self.cfg["drawing"]["show_objects"] = bool(self.chk_show_objects.isChecked())
            self.cfg["drawing"]["show_face_similarity"] = bool(self.cfg["drawing"].get("show_face_similarity", True))
            # Threshold
            if "face" not in self.cfg: self.cfg["face"] = {}
            self.cfg["face"]["similarity_threshold"] = float(self.spin_thresh.value())
            save_config(self.cfg)
            QtWidgets.QMessageBox.information(self, "Sauvé", "Configuration enregistrée.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur de sauvegarde", str(e))

    def on_timer(self):
        if not self.cap: return
        ok, frame = self.cap.read()
        if not ok: return
        self.frame_idx += 1; self.frame = frame
        t0 = time.time()

        # Detection + tracking
        if self.frame_idx % max(1, int(self.cfg["video"]["inference_every_n_frames"])) == 0:
            self.detections = self.detector.detect(frame)
            self.tracked = self.tracker.update(self.detections, class_name='person')

        # Faces
        faces = self.facerec.recognize(
            frame,
            every=max(1,int(self.cfg["face"]["recognize_every_n_frames"])),
            frame_idx=self.frame_idx,
            blur_unknown=bool(self.chk_blur_unknown.isChecked())
        )

        # Persons map
        person_dets = [(d["track_id"], d["box"]) for d in self.tracked if d["name"]=="person" and d.get("track_id")]
        curr_centers: Dict[int, Tuple[int,int]] = {tid: center_of(box) for tid, box in person_dets}

        # Zones
        if len(self.zone_inside_ids) != len(self.zones):
            self.zone_inside_ids = [set() for _ in self.zones]
            self.zone_unique_ids = [set() for _ in self.zones]
        for zi, z in enumerate(self.zones):
            poly = [tuple(p) for p in z.get("polygon", [])]
            current_inside = set()
            for tid, box in person_dets:
                cx, cy = center_of(box)
                if len(poly) >= 3 and point_in_polygon((cx,cy), poly):
                    current_inside.add(tid)
            for tid in list(current_inside):
                if tid not in self.zone_inside_ids[zi]:
                    self.zone_unique_ids[zi].add(tid); self.log_event("zone_enter", tid, z.get("name","Zone"))
            for tid in list(self.zone_inside_ids[zi]):
                if tid not in current_inside:
                    self.log_event("zone_leave", tid, z.get("name","Zone"))
            self.zone_inside_ids[zi] = current_inside

        # Lines
        for ln in self.lines:
            p1 = tuple(ln.get("p1") or ()); p2 = tuple(ln.get("p2") or ())
            if len(p1) != 2 or len(p2) != 2: continue
            counts = ln.setdefault("counts", {"AtoB":0,"BtoA":0})
            for tid, curr in curr_centers.items():
                prev = self.prev_centers.get(tid)
                if not prev: continue
                from .utils import segment_intersection, side_of_line
                if segment_intersection(prev, curr, p1, p2):
                    s_prev = side_of_line(prev, p1, p2); s_curr = side_of_line(curr, p1, p2)
                    if s_prev < 0 and s_curr > 0:
                        counts["AtoB"] += 1; self.log_event("line_cross", tid, ln.get("name","Ligne"), "AtoB")
                    elif s_prev > 0 and s_curr < 0:
                        counts["BtoA"] += 1; self.log_event("line_cross", tid, ln.get("name","Ligne"), "BtoA")
        self.prev_centers = curr_centers

        # Draw detections (persons + objects)
        for det in self.tracked:
            box  = [int(v) for v in det["box"]]
            name = str(det.get("name", ""))
            conf = float(det.get("conf", 0.0))

            # >>> NEW: reconnaître une arme
            is_weapon = (det.get("category") == "weapon") or name.startswith("weapon:")

            # Label
            label = f"{name} {conf:.2f}"
            if det.get("name") == "person" and det.get("track_id"):
                label = f"ID {det['track_id']} - {label}"

            # >>> NEW: ne jamais masquer les armes même si "objets" est décoché
            if not self.chk_show_objects.isChecked():
                if name != "person" and not is_weapon:
                    continue

            # Couleur
            if is_weapon:
                color = (0, 0, 255)  # ROUGE pour armes
                label = label.replace("weapon:", "arme: ")
            elif det.get("name") == "person" and det.get("track_id"):
                color = (0, 255, 0)  # vert pour personnes trackées
            else:
                color = (200, 200, 0)  # jaune pour autres objets

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], max(0, box[1]-6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Draw faces (with similarity)
        for f in faces:
            x1,y1,x2,y2 = [int(v) for v in f["box"]]
            nm = f["name"]; sim = f.get("score", 0.0)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,255),2)
            label = nm + (f" ({sim:.2f})" if bool(self.cfg["drawing"].get("show_face_similarity", True)) else "")
            cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2, cv2.LINE_AA)

        # Draw zones
        zone_colors = [(0,200,255), (255,160,0), (180,255,100), (255,100,180), (150,200,255)]
        for i, z in enumerate(self.zones):
            poly = [tuple(p) for p in z.get("polygon", [])]
            draw_polygon(frame, poly, color=zone_colors[i % len(zone_colors)], closed=len(poly)>=3)
            if poly:
                x = int(np.mean([p[0] for p in poly])); y = int(np.mean([p[1] for p in poly]))
                cv2.putText(frame, z.get("name","Zone"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colors[i % len(zone_colors)], 2, cv2.LINE_AA)

        # Draw lines
        for ln in self.lines:
            p1 = ln.get("p1"); p2 = ln.get("p2")
            if p1 and p2 and len(p1)==2 and len(p2)==2:
                draw_line(frame, tuple(p1), tuple(p2), color=(0,255,255))
                mid = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
                counts = ln.get("counts", {"AtoB":0,"BtoA":0})
                cv2.putText(frame, f'{ln.get("name","Ligne")} A→B:{counts.get("AtoB",0)} B→A:{counts.get("BtoA",0)}',
                            (mid[0]+6, mid[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

        fps = 1.0 / (time.time() - t0 + 1e-6)
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        put_fps(frame, fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video.setPixmap(pix)

        self.refresh_line_list(); self.refresh_tables()

    def closeEvent(self, ev: QtGui.QCloseEvent):
        self.stop_capture()
        return super().closeEvent(ev)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = App(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()