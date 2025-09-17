
import os, time, glob, yaml, math, shutil
import numpy as np
import psutil
from pathlib import Path
import subprocess, re
from ultralytics import YOLO
from gpu_monitor import TegrastatsMonitor
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from collections import defaultdict
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROC = psutil.Process(os.getpid())
PID = os.getpid()



def get_system_usage(device):

    cpu_proc = PROC.cpu_percent(interval=None)
    ram_proc_mb = PROC.memory_info().rss / (1024 ** 2)
    return cpu_proc, ram_proc_mb

def get_tegrastats():
    try:
        out = subprocess.check_output(
            ['tegrastats', '--interval', '1000', '--count', '1'],
            stderr=subprocess.DEVNULL
        )
        line = out.decode('utf-8')
    except Exception:
        return 0.0, 0.0, 0.0

    # GPU %
    gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
    gpu_usage = float(gpu_match.group(1)) if gpu_match else 0.0

    # GPU Power (em mW → W)
    pwr_match = re.search(r'POM_5V_GPU (\d+)mW', line)
    power_watts = float(pwr_match.group(1))/1000 if pwr_match else 0.0

    # RAM usada (MB)
    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    ram_used_mb = float(ram_match.group(1)) if ram_match else 0.0
    # ram_total_mb = float(ram_match.group(2))  # se quiser o total também

    return gpu_usage, power_watts, ram_used_mb


def _format_label(fmt: str) -> str:
    fmt = (fmt or "-").lower()
    if fmt in ["-", "pt", "pytorch"]:
        return "PyTorch"
    if fmt in ["onnx"]:
        return "ONNX"
    if fmt in ["engine", "tensorrt", "trt"]:
        return "TensorRT"
    if fmt in ['deeplab']:
        return 'DeepLab'
    return fmt.upper()

def _export_if_needed(model: YOLO, format: str, device=None):
    print('Verificando necessidade de Exportar o Modelo\n')
    if format.startswith('engine'):
        fmt = 'engine'
    else:
        fmt = format
    
    fmt = (fmt or "-").lower()
    if fmt in ["-", "pt", "pytorch"]:
        return model, None
    save_dir = Path(f'{BASE_DIR}/models')

    if format == 'engineFP16':
        export_path = os.path.join(save_dir, 'land-segFP16.engine')
    elif format == 'engineFP32':
        export_path = os.path.join(save_dir, 'land-segFP32.engine')
    elif format == 'onnx':
        export_path = os.path.join(save_dir, 'land-seg.onnx')
   
    if os.path.exists(export_path):
        model_infer = YOLO(export_path) 
        print('Modelo Já exportado, pulando exportação.')
        return model_infer, export_path

    print(f'Modelo .{fmt} não encontrado, iniciando exportação\n')

    if format == 'engineFP16':
        exported = model.export(format="engine", device=0, half=True, dynamic=False, nms=True)
    else:
        exported = model.export(format=fmt, device=device)
    export_path = None
    

    if isinstance(exported, str) and os.path.exists(exported):
        export_path = exported
    elif hasattr(exported, "save_dir"):
        ext = "engine" if fmt == "engine" else fmt
        cands = glob.glob(os.path.join(exported.save_dir, f"*.{ext}"))
        export_path = cands[0] if cands else None
    

    if format == 'engineFP16':
        new_path = os.path.join(os.path.dirname(export_path), 'land-segFP16.engine')
        shutil.move(export_path, new_path)
        export_path = new_path
    elif format == 'engineFP32':
        new_path = os.path.join(os.path.dirname(export_path), 'land-segFP32.engine')
        shutil.move(export_path, new_path)
        export_path = new_path
    
    model_infer = YOLO(export_path) if export_path else model
    
    return model_infer, export_path

def _model_size_mb(path_or_weights) -> float:
    try:
        if isinstance(path_or_weights, str) and os.path.exists(path_or_weights):
            return round(os.path.getsize(path_or_weights) / (1024 ** 2), 1)
    except Exception:
        pass
    return np.nan

def _load_val_image_paths(data_yaml_path: str):
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    val_entry = cfg.get("val") or cfg.get("val_images") or cfg.get("val_img")
    if val_entry is None:
        raise ValueError("Entrada 'val' não encontrada no data.yml")

    paths = []
    if isinstance(val_entry, str) and os.path.isdir(val_entry):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(val_entry, "**", ext), recursive=True))
    elif isinstance(val_entry, str) and os.path.isfile(val_entry) and val_entry.lower().endswith(".txt"):
        with open(val_entry, "r") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    elif isinstance(val_entry, (list, tuple)):
        paths = list(val_entry)
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(str(val_entry), "**", ext), recursive=True))

    paths = sorted(list({p for p in paths if os.path.isfile(p)}))
    if not paths:
        raise ValueError("Nenhuma imagem de validação encontrada a partir de 'val' em data.yml")
    return paths

def get_val_metrics(yolo_model: YOLO, data, imgsz, device):
    try:
        vr = yolo_model.val(
            data=data, imgsz=imgsz, device=device,
            plots=False, verbose=False, conf=0.001, batch=1
        )
        md = getattr(vr, "results_dict", None) or getattr(getattr(vr, "metrics", None), "results_dict", None) or {}
    except Exception as e:
        print(f"[val] falhou: {type(e).__name__}: {e}")
        return {}

    if not isinstance(md, dict) or not md:
        print("[val] results_dict vazio")
        return {}

    return md 


def compute_metrics(gt_mask, pred_mask):
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(~gt_mask, pred_mask).sum()
    fn = np.logical_and(gt_mask, ~pred_mask).sum()
    tn = np.logical_and(~gt_mask, ~pred_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return precision, recall, iou, f1, accuracy

def warm_up(model, imgsz, device, test_paths='dataset/landslide_dataset_1000/valid/images'):
    print('Aquecendo o modelo\n')
    for _ in range(3):
        for img_path in os.listdir(test_paths)[:5]:  # Usar as primeiras 5 imagens para aquecimento
            _ = model.predict(
                source=os.path.join(test_paths, img_path),
                imgsz=imgsz,
                device=device,
                verbose=False
            )
    print('Aquecimento concluído\n')

def get_metrics_CPU(results=None, img_path=None, fmt_label=None, metrics=None):
    if fmt_label != 'DeepLab':
        masks = results[0].masks.data.cpu().numpy()  # <-- D2H a cada loop
        binary_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
        prediction = binary_mask * 255
    else:
        prediction = results[0].prediction == 1
    
    gt_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0
    prec, rec, iou, f1, acc = compute_metrics(gt, prediction)

    metrics["precision"].append(prec)
    metrics["recall"].append(rec)
    metrics["iou"].append(iou)
    metrics["f1"].append(f1)
    metrics["accuracy"].append(acc)

    # return metrics
        

# ----------------------------
# FUNÇÃO PRINCIPAL 
# ----------------------------
def benchmark_PI(model: str,
                 data: str,
                 imgsz: int = 512,
                 format: str = "-",
                 device=None,
                 limit: int = None):
    status = "✅"
    fmt_label = _format_label(format)
    size_mb = np.nan

    try:
        if fmt_label != 'DeepLab':
            
            base = YOLO(model)
            infer_model, export_artifact = _export_if_needed(base, format, device=device) 
        else:
            print('Usando modelo DeepLab para inferência\n')
            from deep_lab import DeepLab
            deep_lab_path = model.replace('.pt', '.pth')
            print(f'Procurando o modelo DeepLab em {deep_lab_path}')
            if not os.path.exists(deep_lab_path):
                print('Não foi possível encontrar o modelo DeepLab, verifique o caminho e extensão (.pth)')
                raise FileNotFoundError(f'Modelo DeepLab não encontrado em {deep_lab_path}')
            
            infer_model = DeepLab(num_classes=2, model_path=deep_lab_path)
            print('Modelo DeepLab carregado com sucesso\n')
            export_artifact = model

        size_mb = _model_size_mb(export_artifact if export_artifact else model)

        val_paths = _load_val_image_paths(data)
        if limit is not None and isinstance(limit, int) and limit > 0: #Caso seja preciso reduzir o numero de imagens
            val_paths = val_paths[:limit]

        warm_up(infer_model, imgsz, device)
        monitor = None
        if device == 0:
            print(f'Iniciando Monitor_GPU')
            monitor = TegrastatsMonitor(interval_ms=100)  # 10 amostras/s
            monitor.start()
        print(f'Testando com {len(val_paths)} imagens!')
        avg_time = []
        cpu_usages, ram_usages = [], []
        inf_time = 0.0
        metrics = {"iou": [], "precision": [], "recall": [], "f1": [], "accuracy": []}

        total_start = time.time()
        for img_path in val_paths:
            cpu_b, ram_b = get_system_usage(device)
            t0 = time.time()

            results = infer_model.predict(
                source=img_path,
                imgsz=imgsz,
                device=device,
                verbose=False
            )

            
            get_metrics_CPU(results, img_path, fmt_label, metrics)
            # metrics = get_metrics_GPU(results, img_path, fmt_label, metrics)
            inf_time += results[0].speed['inference']  # tempo de inferência em ms
            dt = (time.time() - t0)
            cpu_a, ram_a = get_system_usage(device)

            avg_time.append(dt)
            cpu_usages.append((cpu_b + cpu_a) / 2.0)
            ram_usages.append((ram_b + ram_a) / 2.0)

        total_time = max(1e-9, time.time() - total_start)
        n_imgs = len(avg_time)
        inf_time = inf_time/n_imgs if n_imgs else 0.0

        mean_ms = (np.mean(avg_time) * 1000.0) if n_imgs else np.nan
        fps = (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else 0.0
        throughput = (n_imgs / total_time) if total_time > 0 else 0.0

        ###
        gpu_avg = power_avg = ram_total_used_avg = np.nan
        if monitor:
            gpu_avg, power_avg, ram_total_used_avg = monitor.stop_and_get()

        metrics_yolo = get_val_metrics(infer_model, data=data, imgsz=imgsz, device=device)

        precision = np.mean(metrics["precision"])
        recall = np.mean(metrics["recall"])
        iou = np.mean(metrics["iou"])
        f1 = np.mean(metrics["f1"])
        accuracy = np.mean(metrics["accuracy"])

        row = {
            "Format": fmt_label,
            "Status❔": status,
            "Size (MB)": size_mb,

            # --- desempenho ---
            'Inference Time(ms/im)': round(inf_time, 2) if inf_time == inf_time else np.nan,
            "Avarage Processing Time (ms/im)": round(mean_ms, 2) if mean_ms == mean_ms else np.nan,
            "FPS": round(fps, 2),
            "Throughput (img/s)": round(throughput, 2),

            # --- recursos ---
            "cpu_proc_avg (%)": round(float(np.mean(cpu_usages)) if cpu_usages else np.nan, 2),
            "ram_proc_mb (MB)": round(float(np.mean(ram_usages)) if ram_usages else np.nan, 1),

            "gpu_avg (%)": round(gpu_avg, 2) if gpu_avg==gpu_avg else np.nan,
            "power_avg (W)": round(power_avg, 2) if power_avg==power_avg else np.nan,
            "ram_total_mb (MB)": round(ram_total_used_avg, 1) if ram_total_used_avg==ram_total_used_avg else np.nan,

            # --- métricas de validação ---

            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4)
        }
        # === adicionar a métrica principal dinamicamente ===
        def _r(x, nd=4):
            try:
                v = float(x)
                return np.nan if math.isnan(v) else round(v, nd)
            except Exception:
                return np.nan

        if isinstance(metrics_yolo, dict) and metrics_yolo:
            key = TASK2METRIC.get(infer_model.task, None) 
            print(f'\nA task é {infer_model.task}, métrica principal: {key}\n')
            if key:
                row[key] = _r(metrics_yolo.get(key))

            
            for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
                    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)"):
                if k in metrics_yolo:
                    row[f'{k} (YOLO)'] = _r(metrics_yolo[k])
        else:
            print("[val] sem métricas (checar data.yaml, labels e compatibilidade da task)")

        if format == 'engineFP16':
            row['Format'] = 'TensorRT FP16'
        elif format == 'engineFP32':
            row['Format'] = 'TensorRT FP32'
        return [row]



    except Exception as e:
        return [{
            "Format": _format_label(format),
            "Status❔": f"❌ {type(e).__name__}",
            "Size (MB)": size_mb,
            "metrics/seg/mAP50-95": np.nan,
            "metrics/seg/mAP50": np.nan,
            "metrics/seg/precision": np.nan,
            "metrics/seg/recall": np.nan,
            "metrics/seg/f1": np.nan,
            "metrics/seg/accuracy": np.nan,
            "Inference time (ms/im)": np.nan,
            "FPS": np.nan,
            "Throughput (img/s)": np.nan,
            "cpu_proc_avg (%)": np.nan,
            "ram_proc_mb (MB)": np.nan,
            "gpu_avg (%)": np.nan,
            "vram_avg_mb (MB)": np.nan,
            "vram_proc_mb (MB)": np.nan,
            "power_avg (W)": np.nan,
            "error": str(e),
        }]


