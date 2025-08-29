import os
import glob
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .utils import blur_box

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

class FaceRecognizer:
    def __init__(self, model_name='buffalo_l', provider='cpu', threshold=0.35, facebank_dir='app/data/facebank'):
        providers = ['CPUExecutionProvider'] if provider == 'cpu' else None
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if provider != 'cpu' else -1, det_size=(640, 640))
        self.threshold = float(threshold)
        self.facebank_dir = facebank_dir
        self.gallery = {}
        self.load_facebank()

    def load_facebank(self):
        """Charge les embeddings: privilégie les .npy, sinon recalcule depuis les images."""
        self.gallery = {}
        os.makedirs(self.facebank_dir, exist_ok=True)
        for person in os.listdir(self.facebank_dir):
            pdir = os.path.join(self.facebank_dir, person)
            if not os.path.isdir(pdir):
                continue
            embs = []
            # 1) Embeddings pré-calculés
            for npy in sorted(glob.glob(os.path.join(pdir, '*.npy'))):
                try:
                    arr = np.load(npy)
                    if arr is not None and arr.size > 0:
                        embs.append(arr.astype(np.float32))
                except Exception:
                    pass
            # 2) Fallback: recalcul depuis les images
            if not embs:
                for fn in os.listdir(pdir):
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img = cv2.imread(os.path.join(pdir, fn))
                    if img is None:
                        continue
                    faces = self.app.get(img)
                    if not faces:
                        continue
                    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embs.append(f.normed_embedding.astype(np.float32))
            if embs:
                self.gallery[person] = np.mean(np.stack(embs, axis=0), axis=0)

    def set_threshold(self, thr: float):
        self.threshold = float(thr)

    def recognize(self, frame_bgr, every=3, frame_idx=0, blur_unknown=False):
        if frame_idx % max(1, int(every)) != 0:
            return []
        faces = self.app.get(frame_bgr)
        results = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = f.normed_embedding
            best_name, best_score = "Unknown", -1.0
            for name, g_emb in self.gallery.items():
                sim = cosine_similarity(emb, g_emb)
                if sim > best_score:
                    best_score, best_name = sim, name
            # Acceptation si similarité ≥ seuil
            if best_score < self.threshold:
                best_name = "Unknown"
            if blur_unknown and best_name == "Unknown":
                blur_box(frame_bgr, [x1, y1, x2, y2])
            results.append({"box": [x1, y1, x2, y2], "name": best_name, "score": best_score})
        return results

    def enroll_from_frame(self, frame_bgr, name: str, face_index: int = 0) -> bool:
        """Sauve un crop + l'embedding (.npy) pour 'name' puis recharge la galerie."""
        faces = self.app.get(frame_bgr)
        if not faces:
            return False
        if face_index == 0:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        else:
            if face_index >= len(faces):
                return False
            f = faces[face_index]
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1+1, x2), max(y1+1, y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        pdir = os.path.join(self.facebank_dir, name)
        os.makedirs(pdir, exist_ok=True)
        existing = [fn for fn in os.listdir(pdir) if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]
        out_img = os.path.join(pdir, f"enroll_{len(existing)+1}.jpg")
        cv2.imwrite(out_img, crop)
        # Sauvegarde l'embedding pour rechargement robuste
        try:
            out_npy = out_img.rsplit('.', 1)[0] + '.npy'
            np.save(out_npy, f.normed_embedding.astype(np.float32))
        except Exception:
            pass
        self.load_facebank()
        return True
