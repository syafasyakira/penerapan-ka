"""TrafficVision inference API based on PKA_kepadatan_lalu_lintas notebook."""
import base64
import io
import json
import os
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
DB = ROOT / "traffic.db"
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {"model": "yolo11m.pt", "confidence": 0.10, "imgsz": 1280,
                  "min_area": 600, "iou_threshold": 0.4, "overlap_threshold": 0.6,
                  "q1": None, "q3": None, "roi": [[0, .50], [.60, .20], [.65, .20], [.60, 1], [0, 1]]}
VEHICLES = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}
load_lock = threading.Lock()
model = None
model_name = None


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, source TEXT NOT NULL, total INTEGER NOT NULL, car INTEGER NOT NULL, motorcycle INTEGER NOT NULL, bus INTEGER NOT NULL, truck INTEGER NOT NULL, density TEXT NOT NULL, brightness REAL)")
        conn.commit()


def get_config():
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding='utf-8'))}
        except (ValueError, OSError):
            pass
    return DEFAULT_CONFIG.copy()


app = FastAPI(title="TrafficVision API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()


class FrameRequest(BaseModel):
    image: str
    source: str = "Video"
    save: bool = True


class SettingsRequest(BaseModel):
    confidence: float = Field(ge=0.01, le=0.99)
    imgsz: int = Field(ge=320, le=1920)
    min_area: int = Field(ge=0)
    q1: float | None = None
    q3: float | None = None
    roi: list[list[float]]
    model: str = "yolo11m.pt"


def load_model(filename):
    global model, model_name
    with load_lock:
        if model is None or model_name != filename:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise HTTPException(503, "Ultralytics belum terinstal. Jalankan pip install -r requirements.txt") from e
            try:
                model = YOLO(filename)
                model_name = filename
            except Exception as e:
                raise HTTPException(503, f"Model gagal dimuat: {e}") from e
    return model


def intersection(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    return max(0, x2-x1) * max(0, y2-y1)


def overlap_metrics(a, b):
    inter = intersection(a, b)
    area_a = max(0,a[2]-a[0])*max(0,a[3]-a[1]); area_b = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union = area_a+area_b-inter
    return inter/union if union else 0, inter/min(area_a,area_b) if min(area_a,area_b)>0 else 0


def detect(rgb, config):
    # Same target size / right padding approach used in the notebook.
    h,w = rgb.shape[:2]
    if not h or not w: raise HTTPException(400,"Gambar kosong")
    scale = min(750/h,1333/w)
    resized = cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))))
    canvas = np.zeros((750,1333,3),dtype=np.uint8)
    rh,rw = resized.shape[:2]
    canvas[:rh,:rw] = resized
    roi = np.array([[int(x*1333),int(y*750)] for x,y in config['roi']],dtype=np.int32)
    mask = np.zeros(canvas.shape[:2],dtype=np.uint8)
    cv2.fillPoly(mask,[roi],255)
    gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)
    brightness = round(float(gray[mask>0].mean()),2) if np.any(mask>0) else 0
    detector = load_model(config['model'])
    with load_lock:
        result = detector(canvas, conf=config['confidence'],imgsz=config['imgsz'],classes=list(VEHICLES),verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    candidates=[]
    for b,s,c in zip(boxes,scores,classes):
        if cv2.pointPolygonTest(roi.reshape(-1,1,2),(float((b[0]+b[2])/2),float(b[3])),False)<0:continue
        if (b[2]-b[0])*(b[3]-b[1]) < config['min_area']:continue
        candidates.append((b,float(s),int(c)))
    candidates.sort(key=lambda item:item[1],reverse=True)
    kept=[]
    for b,s,c in candidates:
        if any((lambda m: m[0]>=config['iou_threshold'] or m[1]>=config['overlap_threshold'])(overlap_metrics(b,prev[0])) for prev in kept):
            continue
        kept.append((b,s,c))
    counts = Counter(c for _,_,c in kept)
    total=len(kept)
    q1,q3=config.get('q1'),config.get('q3')
    density='Belum dikalibrasi' if q1 is None or q3 is None else ('Sepi' if total<=q1 else 'Sedang' if total<=q3 else 'Ramai')
    return {"width":1333,"height":750,"boxes":[{"x1":round(float(b[0]),1),"y1":round(float(b[1]),1),"x2":round(float(b[2]),1),"y2":round(float(b[3]),1),"confidence":round(s,3),"class_id":c,"label":VEHICLES[c]} for b,s,c in kept],
            "counts":{"car":counts[2],"motorcycle":counts[3],"bus":counts[5],"truck":counts[7]},"total":total,"density":density,"brightness":brightness,"roi":roi.tolist(),"calibrated":q1 is not None and q3 is not None}


@app.get('/api/health')
def health():
    return {"ok":True,"model_loaded":model is not None,"model":get_config()['model']}


@app.get('/api/settings')
def settings():
    return get_config()


@app.put('/api/settings')
def update_settings(body:SettingsRequest):
    if body.q1 is not None and body.q3 is not None and body.q1>body.q3:
        raise HTTPException(400,'Q1 harus kurang dari atau sama dengan Q3')
    if len(body.roi)<3 or any(len(p)!=2 or any(v<0 or v>1 for v in p) for p in body.roi):
        raise HTTPException(400,'ROI harus berisi minimal 3 titik dengan koordinat 0–1')
    if body.model not in ('yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt'):
        raise HTTPException(400,'Model tidak dikenali')
    config={**get_config(),**body.model_dump()}
    CONFIG_PATH.write_text(json.dumps(config,indent=2),encoding='utf-8')
    return config


@app.post('/api/analyze-frame')
def analyze_frame(payload:FrameRequest):
    try:
        b64=payload.image.split(',',1)[-1]
        if len(b64)>15_000_000: raise ValueError('Frame terlalu besar')
        image=cv2.imdecode(np.frombuffer(base64.b64decode(b64),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Tidak bisa membaca gambar')
    except (ValueError,base64.binascii.Error) as e:
        raise HTTPException(400,f'Format frame tidak valid: {e}') from e
    result=detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),get_config())
    result['timestamp']=datetime.now(timezone.utc).isoformat()
    if payload.save:
        with sqlite3.connect(DB) as conn:
            c=result['counts']
            conn.execute('INSERT INTO observations(timestamp,source,total,car,motorcycle,bus,truck,density,brightness) VALUES (?,?,?,?,?,?,?,?,?)',
                         (result['timestamp'],payload.source[:80],result['total'],c['car'],c['motorcycle'],c['bus'],c['truck'],result['density'],result['brightness']))
    return result


@app.get('/api/observations')
def observations(limit:int=250):
    with sqlite3.connect(DB) as conn:
        conn.row_factory=sqlite3.Row
        rows=conn.execute('SELECT * FROM observations ORDER BY id DESC LIMIT ?', (min(max(limit,1),2000),)).fetchall()
    return [dict(r) for r in rows]


@app.delete('/api/observations')
def clear_observations():
    with sqlite3.connect(DB) as conn: conn.execute('DELETE FROM observations')
    return {"ok":True}


@app.get('/api/stats')
def stats():
    with sqlite3.connect(DB) as conn:
        conn.row_factory=sqlite3.Row
        r=conn.execute('SELECT COUNT(*) n, AVG(total) average, MAX(total) maximum, SUM(car) car, SUM(motorcycle) motorcycle, SUM(bus) bus, SUM(truck) truck FROM observations').fetchone()
        hourly_rows=conn.execute('SELECT timestamp,total FROM observations').fetchall()
    hours={}
    for row in hourly_rows:
        hour=datetime.fromisoformat(row['timestamp']).astimezone(ZoneInfo('Asia/Jakarta')).strftime('%H')
        entry=hours.setdefault(hour,{'hour':hour,'sum':0,'samples':0})
        entry['sum']+=row['total'];entry['samples']+=1
    hourly=[{'hour':item['hour'],'average':round(item['sum']/item['samples'],2),'samples':item['samples']} for item in sorted(hours.values(),key=lambda x:x['hour'])]
    return {"summary":dict(r),"hourly":hourly}
