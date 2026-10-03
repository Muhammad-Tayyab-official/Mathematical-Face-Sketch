from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Stroke:
    points: np.ndarray
    source: str = "edge"
    label: str = "Stroke"


LANDMARK_LOOPS = (
    ((10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109), "Face contour"),
    ((70,63,105,66,107,55,65,52,53,46,70), "Left eyebrow"),
    ((300,293,334,296,336,285,294,282,283,276,300), "Right eyebrow"),
    ((33,7,163,144,145,153,154,155,133,173,157,158,159,160,161), "Left eye"),
    ((362,382,381,380,374,373,390,249,263,466,388,387,386,385,384), "Right eye"),
    ((61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40), "Lips"),
    ((168,6,197,5,4,1,19,94,2,98,97,326,327,294,278), "Nose"),
)


def ensure_bgr(img):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Invalid image.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    raise ValueError("Expected grayscale, BGR or BGRA image.")


def resize_for_processing(img, max_side=1100):
    img = ensure_bgr(img)
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def resample(points, n=100, closed=False):
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        raise ValueError("Not enough points.")
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8
    p = p[keep]
    if len(p) < 2:
        raise ValueError("Not enough unique points.")
    q = np.vstack([p, p[0]]) if closed else p
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(d)]
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError("Zero-length contour.")
    targets = np.linspace(0, total, n, endpoint=not closed)
    return np.c_[np.interp(targets, cum, q[:,0]), np.interp(targets, cum, q[:,1])]


def _mp():
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("MediaPipe is unavailable. Check the pinned environment.") from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        raise RuntimeError("This project requires MediaPipe 0.10.21 with mp.solutions.face_mesh.")
    return mp


def face_geometry(img):
    mp = _mp()
    img = ensure_bgr(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as mesh:
        result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        raise ValueError("No face detected. Use a clear front-facing portrait.")
    face = result.multi_face_landmarks[0]
    h, w = img.shape[:2]
    return face, w, h


def face_strokes(img):
    face, w, h = face_geometry(img)
    out = []
    for ids, label in LANDMARK_LOOPS:
        pts = np.array([[face.landmark[i].x*w, face.landmark[i].y*h] for i in ids], dtype=float)
        if len(pts) >= 3:
            out.append(Stroke(resample(pts, max(80, len(pts)*4), closed=True), "landmark", label))
    return out, face, w, h


def edge_strokes(img, roi=None, max_strokes=55):
    img = ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 55, 145)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    H,W = gray.shape
    candidates=[]
    for c in contours:
        per=float(cv2.arcLength(c, True))
        if per < np.hypot(H,W)*0.025:
            continue
        x,y,w,h=cv2.boundingRect(c)
        if roi is not None:
            rx0,ry0,rx1,ry1=roi
            cx=x+w/2; cy=y+h/2
            if cx<rx0 or cx>rx1 or cy<ry0 or cy>ry1:
                continue
        approx=cv2.approxPolyDP(c, max(0.9, 0.004*per), True)[:,0,:].astype(float)
        if len(approx)<5:
            continue
        area=abs(float(cv2.contourArea(approx.astype(np.float32))))
        if area<30 and per < np.hypot(H,W)*0.06:
            continue
        try:
            pts=resample(approx, min(110,max(35,len(approx)*2)), closed=True)
        except ValueError:
            continue
        candidates.append((per, Stroke(pts,"edge","Image contour")))
    candidates.sort(key=lambda z:z[0], reverse=True)
    return [s for _,s in candidates[:max_strokes]]


def portrait_strokes(img):
    """Feature-first portrait extraction; background edges are constrained to the face/hair region."""
    landmarks, face, w, h = face_strokes(img)
    xs=np.array([p.x*w for p in face.landmark]); ys=np.array([p.y*h for p in face.landmark])
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max()
    pad_x=(x1-x0)*0.42; pad_y=(y1-y0)*0.48
    roi=(max(0,x0-pad_x), max(0,y0-pad_y), min(w,x1+pad_x), min(h,y1+pad_y))
    edges=edge_strokes(img, roi=roi, max_strokes=55)
    return landmarks+edges


def simplify_strokes(strokes, ratio=0.004):
    out=[]
    for s in strokes:
        p=s.points.astype(np.float32).reshape(-1,1,2)
        per=cv2.arcLength(p, True)
        a=cv2.approxPolyDP(p,max(0.5,ratio*per),True)[:,0,:].astype(float)
        if len(a)>=4:
            out.append(Stroke(a,s.source,s.label))
    return out


def normalize_strokes(strokes, span=10.0):
    if not strokes: return []
    pts=np.vstack([s.points for s in strokes])
    xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0)
    scale=2*span/max(float(xmax-xmin),float(ymax-ymin),1.0)
    cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
    out=[]
    for s in strokes:
        q=(s.points-np.array([cx,cy]))*scale
        q[:,1]*=-1
        out.append(Stroke(q,s.source,s.label))
    return out


def _fmt(v,p=3):
    if abs(float(v))<1e-9: v=0
    return f"{float(v):.{p}f}"


def fourier_coefficients(points, harmonics=8):
    """Return real Fourier coefficients for a closed parametric stroke."""
    p=np.asarray(points,float)
    z=p[:,0]+1j*p[:,1]
    c=np.fft.fft(z)/len(z)
    N=len(c)
    kmax=min(harmonics,N//2-1)
    return c[:kmax+1]


def fourier_equation(stroke, harmonics=8, precision=3):
    c=fourier_coefficients(stroke.points,harmonics)
    a0=c[0]
    terms_x=[_fmt(a0.real,precision)]
    terms_y=[_fmt(a0.imag,precision)]
    for k in range(1,len(c)):
        ck=c[k]
        # z(t)=sum c_k exp(i k t) + conjugate symmetry approximation using positive frequencies
        # Using real form from c_k and c_-k is clearer for a displayed equation.
        cn=np.conj(c[-k]) if k < len(c) else 0
        A=2*ck.real; B=-2*ck.imag
        # z contribution: A cos(kt) + B sin(kt) in x, and corresponding y from complex coefficients
        # For a closed real x/y representation, derive directly from sampled coordinate FFTs.
    x=stroke.points[:,0]; y=stroke.points[:,1]
    X=np.fft.fft(x)/len(x); Y=np.fft.fft(y)/len(y)
    kmax=min(harmonics,len(X)//2-1)
    xt=_fmt(X[0].real,precision); yt=_fmt(Y[0].real,precision)
    for k in range(1,kmax+1):
        ax=2*X[k].real; bx=-2*X[k].imag
        ay=2*Y[k].real; by=-2*Y[k].imag
        if abs(ax)>10**(-precision): xt += (" + " if ax>=0 else " - ")+_fmt(abs(ax),precision)+f"cos({k}t)"
        if abs(bx)>10**(-precision): xt += (" + " if bx>=0 else " - ")+_fmt(abs(bx),precision)+f"sin({k}t)"
        if abs(ay)>10**(-precision): yt += (" + " if ay>=0 else " - ")+_fmt(abs(ay),precision)+f"cos({k}t)"
        if abs(by)>10**(-precision): yt += (" + " if by>=0 else " - ")+_fmt(abs(by),precision)+f"sin({k}t)"
    return f"x(t) = {xt}\ny(t) = {yt}\n0 ≤ t ≤ 2π"


def stroke_curve(stroke, samples=180, harmonics=12):
    """Smooth Fourier reconstruction used for animation/display."""
    x=stroke.points[:,0]; y=stroke.points[:,1]
    X=np.fft.fft(x)/len(x); Y=np.fft.fft(y)/len(y)
    kmax=min(harmonics,len(X)//2-1)
    t=np.linspace(0,2*np.pi,samples,endpoint=False)
    xr=np.full(samples,X[0].real); yr=np.full(samples,Y[0].real)
    for k in range(1,kmax+1):
        xr += 2*(X[k].real*np.cos(k*t)-X[k].imag*np.sin(k*t))
        yr += 2*(Y[k].real*np.cos(k*t)-Y[k].imag*np.sin(k*t))
    return np.c_[xr,yr]


def desmos_expression(stroke, precision=4):
    return fourier_equation(stroke, harmonics=8, precision=precision)


def desmos_text(strokes):
    chunks=[]
    for i,s in enumerate(strokes,1):
        chunks.append(f"# {i}. {s.label}\n{fourier_equation(s,8,4)}")
    return "\n\n".join(chunks)
