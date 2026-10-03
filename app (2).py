from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from math_engine import (
    Stroke,
    desmos_expression,
    desmos_text,
    normalize_strokes,
    portrait_strokes,
    resize_for_processing,
    simplify_strokes,
    stroke_curve,
)

st.set_page_config(page_title="Mathematical Portrait", page_icon="∿", layout="wide", initial_sidebar_state="collapsed")

MAX_SIDE = 1100
CANVAS = 760
HARMONICS = 12
SPEED_MULTIPLIER = 2.5
EXPORT_FPS = 30
PIPELINE_VERSION = 4

# Professional, distinct mathematical palette: navy, teal, violet, green,
# amber, terracotta, plum, blue, olive, etc. No neon rainbow.
PALETTE = [
    "#315A7D", "#277A78", "#625C9A", "#3D7C59", "#B07A36",
    "#92576B", "#4D6D9A", "#7A7650", "#6B5B95", "#3F7770",
    "#A06043", "#596F83", "#7B657A", "#50745E", "#8A6A42",
    "#465C82", "#6C6A91", "#567B70", "#9A654F", "#686D4E",
]


def load_image(uploaded):
    pil = Image.open(io.BytesIO(uploaded.getvalue()))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil)
    return resize_for_processing(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), MAX_SIDE)


def bounds(strokes):
    pts = np.vstack([s.points for s in strokes])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def mapper_for(strokes, size=CANVAS, pad=48):
    xmin, ymin, xmax, ymax = bounds(strokes)
    w = max(float(xmax - xmin), 1.0)
    h = max(float(ymax - ymin), 1.0)
    scale = min((size - 2 * pad) / w, (size - 2 * pad) / h)
    ox = (size - w * scale) / 2 - xmin * scale
    oy = (size + h * scale) / 2 + ymin * scale

    def mapper(points):
        q = np.asarray(points, dtype=float)
        return np.column_stack((q[:, 0] * scale + ox, oy - q[:, 1] * scale))
    return mapper


def stroke_length(s):
    return float(np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum())


def color(index):
    return PALETTE[index % len(PALETTE)]


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)


def prepare(strokes):
    # Keep the feature-first ordering. Avoid the previous mistake of sorting
    # every segment by length, which made the portrait lose facial structure.
    strokes = simplify_strokes(strokes, 0.004)
    strokes = normalize_strokes(strokes, span=10.0)
    return strokes[:75]


@st.cache_data(show_spinner=False)
def extract_cached(image_bytes: bytes, version: int = PIPELINE_VERSION):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return prepare(portrait_strokes(img))


def build_payload(strokes):
    payload = []
    for i, s in enumerate(strokes):
        curve = stroke_curve(s, samples=220, harmonics=HARMONICS)
        payload.append({
            "label": s.label,
            "source": s.source,
            "color": color(i),
            "curve": np.round(curve, 4).tolist(),
            "equation": desmos_expression(s, precision=3),
        })
    return payload


def html_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def animation_component(payload, autoplay=True):
    data = json.dumps(payload, separators=(",", ":"))
    html = f"""
<!doctype html>
<html><head><meta charset='utf-8'>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#fff;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#20242a}}
.app{{display:grid;grid-template-columns:minmax(260px,320px) minmax(0,1fr);height:min(760px,78vw);min-height:560px;width:100%;max-width:100%;border:1px solid #e2e5e9;border-radius:12px;overflow:hidden;background:#fff}}
.panel{{border-right:1px solid #e5e7eb;background:#fafbfc;display:flex;flex-direction:column;min-width:0}}
.panel-head{{padding:15px 16px 10px;border-bottom:1px solid #e5e7eb}}
.brand{{font-size:15px;font-weight:650;letter-spacing:-.01em}}.sub{{font-size:11px;color:#7a818a;margin-top:3px}}
.eq-wrap{{padding:10px 10px;overflow:hidden;flex:1}}
.eq-list{{height:100%;overflow:auto;padding-right:4px}}
.eq{{background:#fff;border:1px solid #e5e7eb;border-left:4px solid #d4d8dd;border-radius:8px;padding:9px 9px;margin-bottom:7px;box-shadow:0 1px 1px rgba(0,0,0,.02)}}
.eq.active{{border-color:#d8dce2;background:#fff;box-shadow:0 2px 9px rgba(20,30,40,.07)}}
.eqn{{font-size:10px;color:#8a9199;margin-bottom:5px}}.label{{font-size:10px;color:#535a62;margin-bottom:5px;font-weight:600}}
.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:10px;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:#2e3338}}
.graph{{position:relative;background:#fff;min-width:0;min-height:0}}canvas{{width:100%;height:100%;display:block;background:#fff}}
.overlay{{position:absolute;left:18px;top:15px;pointer-events:none;background:rgba(255,255,255,.88);padding:7px 10px;border:1px solid #e7e9ec;border-radius:7px;font-size:11px;color:#69717a;backdrop-filter:blur(5px)}}
.progress{{height:3px;background:#eceff2;position:absolute;left:0;right:0;bottom:0}}.bar{{height:100%;width:0;background:#526579;transition:width .05s linear}}
@media (max-width: 700px){{
  html,body{{width:100%;max-width:100%;overflow-x:hidden}}
  .app{{display:flex;flex-direction:column;width:100%;height:auto;min-height:0;border-radius:10px}}
  .graph{{order:1;width:100%;height:min(94vw,520px);min-height:300px;aspect-ratio:1/1;flex:0 0 auto}}
  .panel{{order:2;width:100%;height:300px;min-height:300px;border-right:0;border-top:1px solid #e5e7eb}}
  .panel-head{{padding:11px 12px 8px}}
  .brand{{font-size:14px}}
  .sub{{font-size:10px}}
  .eq-wrap{{padding:8px}}
  .eq-list{{padding-right:1px}}
  .eq{{padding:8px 8px;margin-bottom:6px;border-radius:7px}}
  .formula{{font-size:9px;line-height:1.45}}
  .overlay{{left:9px;top:9px;max-width:calc(100% - 18px);font-size:10px;padding:6px 8px}}
}}
@media (min-width: 701px) and (max-width: 900px){{
  .app{{grid-template-columns:280px minmax(0,1fr);height:680px;min-height:0}}
  .formula{{font-size:9px}}
}}
</style></head><body>
<div class='app'>
 <section class='panel'><div class='panel-head'><div class='brand'>Expressions</div><div class='sub'>Live mathematical construction · Fourier parametric strokes</div></div>
 <div class='eq-wrap'><div id='eqList' class='eq-list'></div></div></section>
 <section class='graph'><canvas id='canvas' width='760' height='760'></canvas><div id='overlay' class='overlay'>Equation 1 / {len(payload)}</div><div class='progress'><div id='bar' class='bar'></div></div></section>
</div>
<script>
const strokes={data};
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const eqList=document.getElementById('eqList'), overlay=document.getElementById('overlay'), bar=document.getElementById('bar');
const W=760,H=760,pad=54;
const allPts=strokes.flatMap(s=>s.curve); let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
allPts.forEach(p=>{{xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1]);}});
const scale=Math.min((W-2*pad)/(xmax-xmin||1),(H-2*pad)/(ymax-ymin||1));
const ox=(W-(xmax-xmin)*scale)/2-xmin*scale, oy=(H+(ymax-ymin)*scale)/2+ymin*scale;
function P(p){{return [p[0]*scale+ox,oy-p[1]*scale]}}
function grid(){{ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#f0f2f4';ctx.lineWidth=1;for(let x=30;x<W;x+=38){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}}for(let y=30;y<H;y+=38){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}}ctx.strokeStyle='#e2e5e8';ctx.beginPath();ctx.moveTo(W/2,0);ctx.lineTo(W/2,H);ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke()}}
function drawCurve(s,n,active){{const pts=s.curve; if(n<2)return;ctx.beginPath();let p=P(pts[0]);ctx.moveTo(p[0],p[1]);for(let i=1;i<n;i++){{p=P(pts[i]);ctx.lineTo(p[0],p[1])}}ctx.strokeStyle=active?s.color:'rgba(48,54,61,.74)';ctx.lineWidth=active?3:1.8;ctx.lineJoin='round';ctx.lineCap='round';ctx.stroke();}}
let lastCard=-1;function equationCards(active){{if(active===lastCard)return;lastCard=active;let start=Math.max(0,active-3),end=Math.min(strokes.length,active+4);eqList.innerHTML='';for(let i=start;i<end;i++){{const s=strokes[i],d=document.createElement('div');d.className='eq'+(i===active?' active':'');d.style.borderLeftColor=i===active?s.color:'#d4d8dd';d.innerHTML='<div class="eqn">Equation '+(i+1)+' / '+strokes.length+'</div><div class="label">'+s.label+'</div><div class="formula">'+s.equation+'</div>';eqList.appendChild(d)}}}}
function render(active,frac){{grid();for(let i=0;i<active;i++)drawCurve(strokes[i],strokes[i].curve.length,false);let s=strokes[active];let n=Math.max(2,Math.floor(frac*(s.curve.length-1))+1);drawCurve(s,n,true);let tip=P(s.curve[n-1]);ctx.beginPath();ctx.arc(tip[0],tip[1],10,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(tip[0],tip[1],4,0,Math.PI*2);ctx.fillStyle=s.color;ctx.fill();overlay.textContent='Equation '+(active+1)+' / '+strokes.length+' · '+s.label;bar.style.width=((active+frac)/strokes.length*100)+'%';equationCards(active)}}
let start=null,total=Math.max(2800,Math.min(4600,strokes.length*52))*{SPEED_MULTIPLIER};
// The duration above is normalized so the new trace is about 2.5x faster than the old version.
total=Math.max(2800,Math.min(4600,strokes.length*52));
function animate(ts){{if(start===null)start=ts;let u=Math.min(1,(ts-start)/total);let pos=u*strokes.length;let idx=Math.min(strokes.length-1,Math.floor(pos));let frac=pos-idx;render(idx,frac);if(u<1)requestAnimationFrame(animate);else{{render(strokes.length-1,1);overlay.textContent='Portrait complete · '+strokes.length+' mathematical strokes';bar.style.width='100%'}}}}
grid();equationCards(0);render(0,0);if({str(autoplay).lower()})requestAnimationFrame(animate);
</script></body></html>"""
    return html


@st.cache_data(show_spinner=False)
def make_gif(key, frame_count=150, fps=EXPORT_FPS):
    # Reconstruct the normalized strokes from a compact serialized key.
    strokes=[Stroke(np.asarray(s["points"],float),s["source"],s["label"]) for s in key]
    mapper=mapper_for(strokes)
    frames=[]
    total=frame_count
    for fi in range(total):
        u=fi/max(total-1,1); pos=u*len(strokes); idx=min(len(strokes)-1,int(pos)); frac=pos-idx
        canvas=np.full((CANVAS,CANVAS,3),255,np.uint8)
        # grid
        for x in range(30,CANVAS,38): cv2.line(canvas,(x,0),(x,CANVAS),(240,242,244),1,cv2.LINE_AA)
        for y in range(30,CANVAS,38): cv2.line(canvas,(0,y),(CANVAS,y),(240,242,244),1,cv2.LINE_AA)
        for j,s in enumerate(strokes[:idx+1]):
            pts=stroke_curve(s,220,HARMONICS)
            if j==idx: pts=pts[:max(2,int(frac*len(pts)))]
            q=mapper(pts).round().astype(np.int32)
            if len(q)>1: cv2.polylines(canvas,[q],False,(55,60,66),2,cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)))
    buf=io.BytesIO(); frames[0].save(buf,format='GIF',save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0); return buf.getvalue()


def make_mp4(strokes, fps=EXPORT_FPS, seconds=3.6):
    mapper=mapper_for(strokes); n=max(60,int(fps*seconds));
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'mathematical_portrait.mp4'; writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(CANVAS,CANVAS))
        if not writer.isOpened(): raise RuntimeError('MP4 export is unavailable in this OpenCV build.')
        for fi in range(n):
            u=fi/max(n-1,1); pos=u*len(strokes); idx=min(len(strokes)-1,int(pos)); frac=pos-idx
            canvas=np.full((CANVAS,CANVAS,3),255,np.uint8)
            for x in range(30,CANVAS,38): cv2.line(canvas,(x,0),(x,CANVAS),(240,242,244),1,cv2.LINE_AA)
            for y in range(30,CANVAS,38): cv2.line(canvas,(0,y),(CANVAS,y),(240,242,244),1,cv2.LINE_AA)
            for j,s in enumerate(strokes[:idx+1]):
                pts=stroke_curve(s,220,HARMONICS)
                if j==idx: pts=pts[:max(2,int(frac*len(pts)))]
                q=mapper(pts).round().astype(np.int32); cv2.polylines(canvas,[q],False,(55,60,66),2,cv2.LINE_AA)
            writer.write(canvas)
        writer.release(); return path.read_bytes()


st.markdown("## ∿ Mathematical Portrait Studio")
st.caption("Photo → facial geometry → smooth parametric curves → equations → mathematical portrait")

uploaded=st.file_uploader("Upload portrait",type=["jpg","jpeg","png","webp"],label_visibility="collapsed")
if uploaded is None:
    st.info("Upload a portrait to begin."); st.stop()

try:
    image=load_image(uploaded); ok,enc=cv2.imencode('.png',image)
    if not ok: raise ValueError('Could not encode image.')
    image_bytes=enc.tobytes()
except Exception as exc:
    st.error(f"Could not read image: {exc}"); st.stop()

with st.spinner('Building facial geometry and mathematical curves…'):
    try: strokes=extract_cached(image_bytes)
    except Exception as exc: st.error(str(exc)); st.stop()
if not strokes: st.error('No mathematical strokes were found.'); st.stop()

payload=build_payload(strokes)
st.components.v1.html(animation_component(payload),height=775,scrolling=False)

c1,c2,c3=st.columns([1,1,1])
with c1:
    if st.button('↻ Replay tracing',use_container_width=True):
        st.rerun()
with c2:
    key=tuple({'points':np.round(s.points,6).tolist(),'source':s.source,'label':s.label} for s in strokes)
    try:
        gif=make_gif(key,150,EXPORT_FPS)
        st.download_button('GIF',gif,'mathematical_portrait.gif','image/gif',use_container_width=True)
    except Exception: st.button('GIF',disabled=True,use_container_width=True)
with c3:
    if st.button('MP4',use_container_width=True):
        with st.spinner('Rendering MP4…'):
            try: st.download_button('Download MP4',make_mp4(strokes),'mathematical_portrait.mp4','video/mp4',use_container_width=True)
            except Exception as exc: st.error(f'MP4 export failed: {exc}')

with st.expander('All mathematical equations'):
    st.download_button('Download Desmos equations',desmos_text(strokes).encode(),'portrait_equations.txt','text/plain')
    for i,s in enumerate(strokes):
        st.markdown(f'**{i+1}. {s.label}**')
        st.code(desmos_expression(s,3))
