"""TrafficVision PKA — local ZIP workflow matching PKA_kepadatan_lalu_lintas (2).ipynb.

Importing this module does not load/download YOLO weights. Runs are persisted to disk.
"""
from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import logging
import math
import re
import shutil
import threading
import uuid
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

log = logging.getLogger('trafficvision')
ROOT = Path(__file__).resolve().parent
STORE = ROOT / 'data' / 'runs'
STORE.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'yolo11m.pt'
TARGET_W, TARGET_H = 1333, 750
CONF = 0.10
IMGSZ = 1280
MIN_AREA = 600
IOU_THRESHOLD = 0.4
OVERLAP_THRESHOLD = 0.6
VEHICLES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
ROI_RELATIVE = [(0.00, 0.50), (0.60, 0.20), (0.65, 0.20), (0.60, 1.00), (0.00, 1.00)]
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
TIMESTAMP_FMT = '%d_%m_%Y_%H-%M-%S'
MAX_ZIP_BYTES = 500 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGES = 3000
MAX_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 45_000_000

app = FastAPI(title='TrafficVision PKA — ZIP Dataset Analyzer', version='2.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])
current_model = None
model_lock = threading.Lock()
run_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise HTTPException(400, 'ID analisis tidak valid')
    p = STORE / run_id
    if not p.exists():
        raise HTTPException(404, 'Analisis tidak ditemukan')
    return p


def write_json(path: Path, data: Any):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_manifest(p: Path):
    try:
        return json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(404, 'Data analisis tidak tersedia')


def load_yolo():
    global current_model
    with model_lock:
        if current_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError('Ultralytics belum terpasang. Jalankan python -m pip install -r requirements.txt di folder backend.') from exc
            current_model = YOLO(MODEL_NAME)
    return current_model


def resize_with_right_padding(img: Image.Image) -> Image.Image:
    """Same operations and integer rounding as notebook cell 12."""
    w, h = img.size
    scale = TARGET_H / h
    new_w, new_h = int(w * scale), TARGET_H
    if new_w <= 0:
        raise ValueError('Gambar tidak valid')
    img = img.resize((new_w, new_h))
    if new_w > TARGET_W:
        scale = TARGET_W / new_w
        new_w, new_h = TARGET_W, int(new_h * scale)
        img = img.resize((new_w, new_h))
    canvas = Image.new('RGB', (TARGET_W, TARGET_H), 'black')
    canvas.paste(img, (0, 0))
    return canvas


def get_roi():
    return [(int(x * TARGET_W), int(y * TARGET_H)) for x, y in ROI_RELATIVE]


def intersection(box1, box2):
    x1 = max(box1[0],box2[0]); y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2]); y2 = min(box1[3],box2[3])
    return max(0, x2-x1) * max(0,y2-y1)


def calculate_iou(box1, box2):
    inter = intersection(box1,box2)
    a = (box1[2]-box1[0])*(box1[3]-box1[1]); b = (box2[2]-box2[0])*(box2[3]-box2[1])
    return float(inter / (a+b-inter)) if a+b-inter > 0 else 0


def overlap_ratio(box1,box2):
    inter = intersection(box1,box2)
    a = (box1[2]-box1[0])*(box1[3]-box1[1]); b = (box2[2]-box2[0])*(box2[3]-box2[1])
    return float(inter / min(a,b)) if min(a,b)>0 else 0


def detect_image(img: Image.Image, model):
    arr = np.asarray(resize_with_right_padding(img.convert('RGB')))
    roi_points = np.array(get_roi(), dtype=np.int32)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask,[roi_points],255)
    gray = cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    brightness = float(gray[mask > 0].mean())
    # YOLO runs on FULL image; ROI only filters predictions afterwards (notebook cell 21).
    with model_lock:
        result = model(arr, conf=CONF, imgsz=IMGSZ, classes=list(VEHICLES), verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    if len(boxes):
        contour = roi_points.reshape(-1,1,2)
        inside = np.array([cv2.pointPolygonTest(contour,(float((b[0]+b[2])/2),float(b[3])),False)>=0 for b in boxes])
        boxes,scores,classes = boxes[inside],scores[inside],classes[inside]
    if len(boxes):
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        valid = areas >= MIN_AREA
        boxes,scores,classes = boxes[valid],scores[valid],classes[valid]
    # Identical notebook greedy overlap filtering; suppression is class-agnostic.
    order = np.argsort(scores)[::-1]
    remove = set()
    for i in range(len(order)):
        idx1 = int(order[i])
        if idx1 in remove:
            continue
        for j in range(i+1,len(order)):
            idx2 = int(order[j])
            if idx2 in remove:
                continue
            if calculate_iou(boxes[idx1],boxes[idx2]) >= IOU_THRESHOLD or overlap_ratio(boxes[idx1],boxes[idx2]) >= OVERLAP_THRESHOLD:
                remove.add(idx2)
    counts={name:0 for name in VEHICLES.values()}
    annotations=[]
    for i,(box,conf,cls) in enumerate(zip(boxes,scores,classes)):
        removed=i in remove
        if not removed: counts[VEHICLES[int(cls)]]+=1
        annotations.append({'bbox':[round(float(x),2) for x in box], 'class':VEHICLES[int(cls)], 'confidence':round(float(conf),4),'removed':removed})
    return {'counts':counts,'total_vehicle':sum(counts.values()),'brightness':round(brightness,2), 'annotations':annotations,'keep':len(annotations)-len(remove),'remove':len(remove)}



def detect_vehicles(image_path: Path, model):
    with Image.open(image_path) as img:
        return detect_image(img, model)


class LiveFrame(BaseModel):
    image: str = Field(max_length=14_000_000)
    run_id: str = Field(min_length=32, max_length=32)
    source: str = Field(default='video', pattern='^(video|camera)$')


@app.post('/api/monitor/analyze')
def analyze_monitor_frame(payload: LiveFrame):
    """Analyze one browser-sampled frame using the same pipeline as dataset images.

    Q1/Q3 are NEVER recalculated here; load them from an already completed dataset run.
    Video bytes are not stored on the server.
    """
    p = run_dir(payload.run_id)
    manifest = read_manifest(p)
    if manifest['status'] != 'completed':
        raise HTTPException(409, 'Selesaikan analisis ZIP agar Q1/Q3 tersedia.')
    try:
        b64 = payload.image.split(',', 1)[-1]
        content = base64.b64decode(b64, validate=True)
        if len(content) > 8_000_000:
            raise ValueError('Frame lebih besar dari 8 MB')
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            if source.width * source.height > 45_000_000:
                raise ValueError('Resolusi frame terlalu besar')
            det = detect_image(source, load_yolo())
    except (ValueError, UnidentifiedImageError, binascii.Error, OSError) as exc:
        raise HTTPException(400, f'Frame tidak valid: {exc}') from exc
    thresholds = json.loads((p / 'thresholds.json').read_text(encoding='utf-8'))
    total = det['total_vehicle']
    q1, q3 = thresholds['q1'], thresholds['q3']
    density = 'Sepi' if total <= q1 else ('Sedang' if total <= q3 else 'Ramai')
    return {'timestamp': datetime.now().isoformat(timespec='seconds'),
            'source': payload.source, 'model': MODEL_NAME,
            'threshold_run_id': payload.run_id, 'q1': q1, 'q3': q3,
            'width': TARGET_W, 'height': TARGET_H, 'roi': get_roi(),
            'density': density, **det}

def quantile_linear(values, p):
    return float(np.quantile(np.asarray(values,dtype=np.float64),p))


def summarize(rows):
    """Notebook cells 29–32: Q1/Q3 on all images; hours with count>=3; top 3 mean."""
    totals=[int(r['total_vehicle']) for r in rows]
    q1=quantile_linear(totals,.25); q3=quantile_linear(totals,.75)
    for row in rows:
        v=row['total_vehicle']
        row['density']='Sepi' if v<=q1 else ('Sedang' if v<=q3 else 'Ramai')
    groups={}
    for row in rows: groups.setdefault(row['hour'],[]).append(row['total_vehicle'])
    hourly=[{'hour':hour,'mean':float(np.mean(counts)),'count':len(counts),'total':sum(counts)} for hour,counts in sorted(groups.items())]
    eligible=[h for h in hourly if h['count']>=3]
    peak=sorted(eligible,key=lambda r:-r['mean'])[:3]
    thresholds={'q1':q1,'q3':q3,'peak_hours':[int(h['hour']) for h in peak], 'hourly_mean':{str(h['hour']):round(h['mean'],2) for h in eligible}}
    distribution=Counter(r['density'] for r in rows)
    return thresholds, hourly, {'Sepi':distribution['Sepi'],'Sedang':distribution['Sedang'],'Ramai':distribution['Ramai']}


def save_csv(path,rows):
    cols=['image','timestamp','date','hour','brightness','conf','car','motorcycle','bus','truck','total_vehicle','density']
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def process_run(run_id):
    path=STORE/run_id
    manifest=read_manifest(path)
    manifest['status']='processing'
    write_json(path/'manifest.json',manifest)
    try:
        model=load_yolo()
        rows=[]
        metadata=manifest['images']
        for idx,item in enumerate(metadata):
            try:
                det=detect_vehicles(path/'images'/item['stored'],model)
                row={'id':item['id'],'image':item['name'],'timestamp':item['timestamp'],'date':item['date'],'hour':item['hour'],
                     'brightness':det['brightness'],'conf':CONF,**det['counts'],'total_vehicle':det['total_vehicle'],'keep':det['keep'],'remove':det['remove']}
                rows.append(row)
                write_json(path/'detections'/f"{item['id']}.json",det)
            except Exception as err:
                manifest['skipped'].append({'image':item['name'],'reason':f'Gagal memproses: {err}'})
                log.exception('Image failed: %s', item['name'])
            manifest['processed']=idx+1
            manifest['successful']=len(rows)
            manifest['current_image']=item['name']
            write_json(path/'manifest.json',manifest)
        if not rows: raise ValueError('Tidak ada gambar yang berhasil dianalisis.')
        rows.sort(key=lambda r:r['timestamp'])
        thresholds,hourly,distribution=summarize(rows)
        write_json(path/'rows.json',rows)
        write_json(path/'thresholds.json',thresholds)
        write_json(path/'analytics.json',{'hourly':hourly,'distribution':distribution})
        save_csv(path/'vehicle_counting.csv',rows)
        manifest.update(status='completed',processed=len(metadata),successful=len(rows),current_image=None,finished_at=datetime.now().isoformat(timespec='seconds'))
    except Exception as exc:
        manifest.update(status='failed',error=str(exc),current_image=None)
        log.exception('Run failed')
    finally:
        write_json(path/'manifest.json',manifest)


@app.get('/api/health')
def health():
    return {'ok':True,'model':MODEL_NAME,'model_loaded':current_model is not None}


@app.get('/api/methodology')
def methodology():
    return {'model':MODEL_NAME,'confidence':CONF,'imgsz':IMGSZ,'min_area':MIN_AREA,'iou_threshold':IOU_THRESHOLD,'overlap_threshold':OVERLAP_THRESHOLD,'target_width':TARGET_W,'target_height':TARGET_H,'roi':get_roi(),'timestamp_format':TIMESTAMP_FMT,'peak_min_samples':3,'labels':VEHICLES}


@app.get('/api/runs')
def runs():
    found=[]
    for p in STORE.iterdir():
        if p.is_dir() and (p/'manifest.json').exists():
            try:
                m=read_manifest(p)
                found.append({k:m.get(k) for k in ('id','filename','created_at','status','total_images','processed','successful')})
            except HTTPException: pass
    return sorted(found,key=lambda m:m['created_at'] or '',reverse=True)


@app.post('/api/runs',status_code=201)
async def create_run(file: UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith('.zip'):
        raise HTTPException(400,'Pilih file dataset .zip')
    with run_lock:
        for p in STORE.iterdir():
            if p.is_dir() and (p/'manifest.json').exists():
                m=read_manifest(p)
                if m.get('status') in ('processing','queued'):
                    raise HTTPException(409,'Masih ada dataset yang sedang diproses. Selesaikan dahulu.')
        rid=uuid.uuid4().hex
        path=STORE/rid
        (path/'images').mkdir(parents=True)
        (path/'detections').mkdir()
        zippath=path/'dataset.zip'
        size=0
        try:
            with zippath.open('wb') as out:
                while chunk:=await file.read(1024*1024):
                    size+=len(chunk)
                    if size>MAX_ZIP_BYTES: raise HTTPException(413,'ZIP terlalu besar (maksimal 500 MB)')
                    out.write(chunk)
            metadata=[];skipped=[]
            with zipfile.ZipFile(zippath) as z:
                entries=[x for x in z.infolist() if not x.is_dir() and Path(x.filename).suffix.lower() in EXTS and not x.filename.startswith('__MACOSX/')]
                if len(entries)>MAX_IMAGES: raise HTTPException(413,f'Maksimal {MAX_IMAGES} gambar per ZIP')
                if sum(e.file_size for e in entries)>MAX_UNCOMPRESSED:raise HTTPException(413,'Ukuran ekstraksi ZIP terlalu besar')
                for entry in entries:
                    name=Path(entry.filename).name
                    try:stamp=datetime.strptime(Path(name).stem,TIMESTAMP_FMT)
                    except ValueError:
                        skipped.append({'image':name,'reason':'Nama file harus DD_MM_YYYY_HH-MM-SS.jpg/png/webp'});continue
                    if entry.file_size>MAX_IMAGE_BYTES:
                        skipped.append({'image':name,'reason':'Gambar terlalu besar'});continue
                    try:
                        image_bytes=z.read(entry)
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            im.verify()
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            if im.width * im.height > 45_000_000:raise ValueError('Resolusi gambar terlalu besar')
                            image_width,image_height=im.size
                        idx=len(metadata)+1; stored=f'{idx:06d}{Path(name).suffix.lower()}'
                        (path/'images'/stored).write_bytes(image_bytes)
                        metadata.append({'id':idx,'name':name,'stored':stored,'timestamp':stamp.strftime('%Y-%m-%d %H:%M:%S'),'date':stamp.date().isoformat(),'hour':stamp.hour,'width':image_width,'height':image_height})
                    except Exception as exc:
                        skipped.append({'image':name,'reason':f'Gambar tidak valid: {exc}'})
            if not metadata:raise HTTPException(400,'ZIP tidak memuat gambar valid dengan nama DD_MM_YYYY_HH-MM-SS.jpg')
            hour_count=dict(sorted(Counter(str(x['hour']) for x in metadata).items(),key=lambda t:int(t[0])))
            size_count=Counter(f"{x['width']}×{x['height']}" for x in metadata)
            eda={'hour_count':hour_count,'image_sizes':[{'resolution':key,'count':v} for key,v in size_count.most_common(10)]}
            manifest={'id':rid,'eda':eda,'filename':Path(file.filename).name,'created_at':datetime.now().isoformat(timespec='seconds'),
                      'status':'queued','total_images':len(metadata),'processed':0,'successful':0,'current_image':None,'error':None,'images':metadata,'skipped':skipped}
            write_json(path/'manifest.json',manifest)
            # The original upload is no longer needed after secure extraction.
            zippath.unlink(missing_ok=True)
            threading.Thread(target=process_run,args=(rid,),daemon=True,name=f'run-{rid[:8]}').start()
        except Exception:
            shutil.rmtree(path,ignore_errors=True)
            raise
        finally:
            await file.close()
    return {'id':rid,'status':'queued','total_images':len(metadata),'skipped':len(skipped)}


@app.get('/api/runs/{run_id}')
def run_detail(run_id:str):
    m=read_manifest(run_dir(run_id))
    return {k:v for k,v in m.items() if k!='images'}


@app.get('/api/runs/{run_id}/results')
def run_results(run_id:str):
    p=run_dir(run_id)
    m=read_manifest(p)
    if m['status']!='completed':raise HTTPException(409,'Analisis belum selesai')
    return {'run':{k:v for k,v in m.items() if k!='images'},
            'rows':json.loads((p/'rows.json').read_text(encoding='utf-8')),
            'thresholds':json.loads((p/'thresholds.json').read_text(encoding='utf-8')),
            'analytics':json.loads((p/'analytics.json').read_text(encoding='utf-8'))}


@app.get('/api/runs/{run_id}/image/{image_id}')
def run_image(run_id:str,image_id:int):
    p=run_dir(run_id)
    m=read_manifest(p)
    item=next((i for i in m['images'] if i['id']==image_id),None)
    if not item:raise HTTPException(404,'Gambar tidak ditemukan')
    with Image.open(p/'images'/item['stored']) as img:
        image=resize_with_right_padding(img.convert('RGB'))
        buf=io.BytesIO();image.save(buf,format='JPEG',quality=82)
    return Response(content=buf.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=3600'})


@app.get('/api/runs/{run_id}/detections/{image_id}')
def run_detections(run_id:str,image_id:int):
    p=run_dir(run_id)
    file=p/'detections'/f'{image_id}.json'
    if not file.exists():raise HTTPException(404,'Detail deteksi belum tersedia')
    return json.loads(file.read_text(encoding='utf-8'))


@app.get('/api/runs/{run_id}/download/{kind}')
def download_result(run_id:str,kind:str):
    p=run_dir(run_id)
    filename={'csv':'vehicle_counting.csv','thresholds':'thresholds.json','analytics':'analytics.json'}.get(kind)
    if not filename:raise HTTPException(404,'Format unduhan tidak dikenal')
    if not (p/filename).exists():raise HTTPException(409,'Hasil belum tersedia')
    return FileResponse(p/filename,filename=filename)
