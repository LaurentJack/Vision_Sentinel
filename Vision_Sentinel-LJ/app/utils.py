from typing import List, Tuple
import cv2
import numpy as np
try:
    from shapely.geometry import Point, Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

Color = Tuple[int, int, int]

def put_fps(frame, fps: float, org=(10, 30), color=(255,255,255)):
    cv2.putText(frame, f"FPS: {fps:.1f}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def draw_polygon(frame, points: List[Tuple[int,int]], color: Color=(0, 200, 255), closed: bool=False):
    if not points:
        return
    for i in range(1, len(points)):
        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, 2)
    if closed and len(points) > 2:
        cv2.line(frame, tuple(points[-1]), tuple(points[0]), color, 2)

def draw_line(frame, p1: Tuple[int,int], p2: Tuple[int,int], color: Color=(0, 255, 255)):
    cv2.line(frame, p1, p2, color, 2)
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv2.circle(frame, mid, 4, color, -1)

def point_in_polygon(point: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    if _HAS_SHAPELY:
        return Polygon(poly).contains(Point(point[0], point[1]))
    # fallback ray casting
    x, y = point
    inside = False
    p1x, p1y = poly[0]
    for i in range(len(poly)+1):
        p2x, p2y = poly[i % len(poly)]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def side_of_line(p, a, b) -> float:
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def segment_intersection(p1, p2, q1, q2) -> bool:
    def ccw(a,b,c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and (ccw(p1,p2,q1) != ccw(p1,p2,q2))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union

def center_of(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def draw_transparent_rect(img, box, color=(0, 255, 0), alpha=0.2):
    overlay = img.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def blur_box(img, box):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return
    face = cv2.GaussianBlur(face, (31,31), 0)
    img[y1:y2, x1:x2] = face

import numpy as np
import cv2

def _hue_to_name(h: int, s: float, v: float) -> str:
    # h ∈ [0..179], s,v ∈ [0..1]
    if h < 10 or h >= 170: base = "rouge"
    elif h < 25:           base = "orange"
    elif h < 35:           base = "jaune"
    elif h < 85:           base = "vert"
    elif h < 95:           base = "cyan"
    elif h < 130:          base = "bleu"
    elif h < 150:          base = "violet"
    else:                  base = "rose"
    # marron (brun) = orange/jaune sombre
    if base in ("orange","jaune") and v < 0.60 and s > 0.30:
        return "marron"
    return base

def _achromatic_name(s_mean: float, v_mean: float) -> str:
    # niveaux achromatiques (noir / gris / blanc)
    if v_mean < 0.25: return "noir"
    if v_mean > 0.85 and s_mean < 0.25: return "blanc"
    return "gris"

def dominant_color_name_bgr(bgr_crop: np.ndarray) -> str:
    """Retourne un nom de couleur pour une région BGR."""
    if bgr_crop is None or bgr_crop.size == 0:
        return ""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0
    h = h.reshape(-1); s = s.reshape(-1); v = v.reshape(-1)

    # pixels "chromatiques"
    mask = (s > 0.25) & (v > 0.20) & (v < 0.95)
    if np.count_nonzero(mask) > 50:
        hh = h[mask].astype(np.int32)
        hist = np.bincount(hh, minlength=180)
        peak = int(hist.argmax())
        return _hue_to_name(peak, float(s[mask].mean()), float(v[mask].mean()))
    # sinon: achromatique
    return _achromatic_name(float(s.mean()), float(v.mean()))

def infer_color_for_box(frame_bgr: np.ndarray, box, focus_torso: bool = False) -> str:
    """Estime la couleur dominante dans une box. Si focus_torso=True, ne garde que le torse."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(1, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1: 
        return ""
    crop = frame_bgr[y1:y2, x1:x2]

    if focus_torso and crop.size > 0:
        ch, cw = crop.shape[:2]
        # fenêtre "torse" ~ milieu du cadre
        tx1 = int(0.25 * cw); tx2 = int(0.75 * cw)
        ty1 = int(0.35 * ch); ty2 = int(0.85 * ch)
        tx1 = max(0, min(cw-1, tx1)); tx2 = max(1, min(cw, tx2))
        ty1 = max(0, min(ch-1, ty1)); ty2 = max(1, min(ch, ty2))
        crop = crop[ty1:ty2, tx1:tx2]
    return dominant_color_name_bgr(crop)
