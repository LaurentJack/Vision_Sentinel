from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """
    Détecteur combiné :
    - Détection générale (COCO, personnes/objets)
    - Détection 'weapons' optionnelle (poids custom), fusionnée dans la sortie
    """
    def __init__(self, model_name='yolov8n.pt', classes_of_interest=None, conf=0.35, iou=0.5,
                 pose_model_name=None, pose_every=1,
                 weapons_enabled=False, weapons_model=None, weapons_conf=0.45, weapons_iou=0.5,
                 weapons_aliases=None):
        # Détection générale
        self.model = YOLO(model_name)
        self.names = self.model.names
        self.class_filter = set(classes_of_interest) if classes_of_interest else None
        self.conf = conf
        self.iou = iou

        # Pose optionnelle (si tu avais déjà intégré la pose)
        self.pose_model = YOLO(pose_model_name) if pose_model_name else None
        self.pose_every = max(1, int(pose_every))
        self._frame_idx = 0

        # Détection armes (second modèle)
        self.weapons_enabled = bool(weapons_enabled and weapons_model)
        self.wmodel = YOLO(weapons_model) if self.weapons_enabled else None
        self.wconf = weapons_conf
        self.wiou = weapons_iou
        self.waliases = weapons_aliases or {}

    def _filter_class(self, name: str) -> bool:
        return (self.class_filter is None) or (name in self.class_filter)

    def detect(self, frame_bgr) -> List[Dict[str, Any]]:
        self._frame_idx += 1
        dets: List[Dict[str, Any]] = []

        # 1) Détection générale
        g = self.model.predict(source=frame_bgr, verbose=False, conf=self.conf, iou=self.iou, imgsz=640)
        if g:
            res = g[0]; boxes = res.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    b = boxes[i]
                    cls_id = int(b.cls[0].item()); conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    name = self.model.names.get(cls_id, str(cls_id))
                    if self._filter_class(name):
                        dets.append({"class_id": cls_id, "name": name, "conf": conf, "box": xyxy})

        # 2) Pose optionnelle (si présente) → attacher keypoints aux 'person'
        if self.pose_model and (self._frame_idx % self.pose_every == 0):
            pres = self.pose_model.predict(source=frame_bgr, verbose=False, conf=self.conf, iou=self.iou, imgsz=640)
            if pres:
                p0 = pres[0]; pboxes = p0.boxes; pk = p0.keypoints
                if pboxes is not None and pk is not None:
                    for i in range(len(pboxes)):
                        p_box = pboxes[i].xyxy[0].cpu().numpy().tolist()
                        kxy = pk.xy[i].cpu().numpy()  # (17,2)
                        if hasattr(pk, 'conf') and pk.conf is not None:
                            kconf = pk.conf[i].cpu().numpy()
                        else:
                            kconf = np.ones((kxy.shape[0],), dtype=np.float32)
                        kpts = [(float(x), float(y), float(c)) for (x, y), c in zip(kxy, kconf)]
                        # associer à la meilleure personne par IoU
                        def iou(a,b):
                            xA=max(a[0],b[0]); yA=max(a[1],b[1]); xB=min(a[2],b[2]); yB=min(a[3],b[3])
                            inter=max(0,xB-xA)*max(0,yB-yA)
                            areaA=max(0,a[2]-a[0])*max(0,a[3]-a[1]); areaB=max(0,b[2]-b[0])*max(0,b[3]-b[1])
                            return inter/(areaA+areaB-inter+1e-9)
                        best=None; best_iou=0.0
                        for d in dets:
                            if d["name"]!="person": continue
                            s=iou(d["box"], p_box)
                            if s>best_iou: best_iou=s; best=d
                        if best is not None and best_iou>=0.30:
                            best["keypoints"] = kpts

        # 3) Détection d'armes (seulement si activée)
        if self.weapons_enabled and self.wmodel is not None:
            w = self.wmodel.predict(source=frame_bgr, verbose=False, conf=self.wconf, iou=self.wiou, imgsz=640)
            if w:
                wr = w[0]; wboxes = wr.boxes
                if wboxes is not None:
                    # Si le modèle a ses propres noms :
                    wnames = getattr(self.wmodel, "names", {})
                    for i in range(len(wboxes)):
                        b = wboxes[i]
                        cls_id = int(b.cls[0].item()); conf = float(b.conf[0].item())
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        raw = wnames.get(cls_id, str(cls_id))
                        name = self.waliases.get(raw, raw)  # harmonisation
                        dets.append({
                            "class_id": cls_id,
                            "name": f"weapon:{name}",
                            "category": "weapon",
                            "conf": conf,
                            "box": xyxy
                        })

        return dets
