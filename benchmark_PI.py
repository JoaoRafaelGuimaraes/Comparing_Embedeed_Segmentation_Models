# -*- coding: utf-8 -*-
import os, time, glob, yaml, math
import numpy as np
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

from ultralytics import YOLO

# ----------------------------
# GPU/CPU monitor (por processo)
# ----------------------------
def _get_gpu_handle_or_none(index=0):
    if not _NVML_OK:
        return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None

_GPU_HANDLE = _get_gpu_handle_or_none(0)
PROC = psutil.Process(os.getpid())
PID = os.getpid()

def get_system_usage():
    """
    Retorna (cpu_proc%, ram_proc_MB, gpu_global%, vram_global_MB, power_W, vram_proc_MB).
    - CPU/RAM: do processo atual
    - GPU/power: globais (placa inteira)
    - VRAM_proc: VRAM usada por ESTE processo (se disponível)
    """
    # CPU% do processo; para leituras estáveis, esta função deve ser chamada em momentos diferentes
    cpu_proc = PROC.cpu_percent(interval=None)
    ram_proc_mb = PROC.memory_info().rss / (1024 ** 2)

    gpu_usage = 0.0
    vram_usage_mb = 0.0
    power_watts = 0.0
    vram_proc_mb = 0.0

    if _GPU_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            power = pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0

            gpu_usage = float(util.gpu)                      # % global da GPU
            vram_usage_mb = float(mem.used) / (1024 ** 2)    # VRAM global (MB)
            power_watts = float(power)                       # potência global (W)

            # VRAM do processo atual (MB)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_GPU_HANDLE)
                for p in procs:
                    if getattr(p, "pid", None) == PID:
                        vram_proc_mb = float(p.usedGpuMemory) / (1024 ** 2)
                        break
            except Exception:
                pass
        except Exception:
            pass

    return cpu_proc, ram_proc_mb, gpu_usage, vram_usage_mb, power_watts, vram_proc_mb


# ----------------------------
# Helpers
# ----------------------------
def _format_label(fmt: str) -> str:
    fmt = (fmt or "-").lower()
    if fmt in ["-", "pt", "pytorch"]:
        return "PyTorch"
    if fmt in ["onnx"]:
        return "ONNX"
    if fmt in ["engine", "tensorrt", "trt"]:
        return "TensorRT"
    return fmt.upper()

def _export_if_needed(model: YOLO, fmt: str, device=None):
    """Exporta se fmt != '-' e retorna (modelo_infer, caminho_artefato_ou_None)."""
    fmt = (fmt or "-").lower()
    if fmt in ["-", "pt", "pytorch"]:
        return model, None
    exported = model.export(format=fmt, device=device)
    export_path = None
    if isinstance(exported, str) and os.path.exists(exported):
        export_path = exported
    elif hasattr(exported, "save_dir"):
        ext = "engine" if fmt == "engine" else fmt
        cands = glob.glob(os.path.join(exported.save_dir, f"*.{ext}"))
        export_path = cands[0] if cands else None
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

def _safe_val_metrics_seg(yolo_model: YOLO, data, imgsz, device):
    """
    Tenta extrair métricas de SEGMENTAÇÃO do model.val().
    Retorna dicionário com chaves normalizadas para máscara.
    """
    try:
        vr = yolo_model.val(data=data, imgsz=imgsz, device=device, verbose=False, plots=False)
        if hasattr(vr, "results_dict") and isinstance(vr.results_dict, dict):
            md = vr.results_dict
        elif hasattr(vr, "metrics") and hasattr(vr.metrics, "results_dict"):
            md = vr.metrics.results_dict
        else:
            md = {}
    except Exception:
        md = {}

    out = {}

    seg_map_keys = [
        "metrics/seg/mAP50-95",
        "metrics/mAP50-95(M)",
        "metrics/mAP50-95(seg)",
    ]
    for k in seg_map_keys:
        if k in md:
            out["metrics/seg/mAP50-95"] = md[k]
            break

    for k in ["metrics/seg/mAP50", "metrics/mAP50(M)"]:
        if k in md:
            out["metrics/seg/mAP50"] = md[k]
            break

    for src in ["metrics/seg/precision", "metrics/precision(M)", "metrics/precision"]:
        if src in md:
            out["metrics/seg/precision"] = md[src]
            break
    for src in ["metrics/seg/recall", "metrics/recall(M)", "metrics/recall"]:
        if src in md:
            out["metrics/seg/recall"] = md[src]
            break
    for src in ["metrics/seg/f1", "metrics/f1(M)", "metrics/f1"]:
        if src in md:
            out["metrics/seg/f1"] = md[src]
            break

    for src in ["metrics/seg/accuracy", "metrics/accuracy", "metrics/acc"]:
        if src in md:
            out["metrics/seg/accuracy"] = md[src]
            break

    if "metrics/seg/mAP50-95" not in out:
        for k in ["metrics/mAP50-95(M)", "metrics/mAP50-95"]:
            if k in md:
                out["metrics/seg/mAP50-95"] = md[k]
                break

    return out


# ----------------------------
# FUNÇÃO PRINCIPAL (SEGMENTAÇÃO)
# ----------------------------
def benchmark_PI(model: str,
                 data: str,
                 imgsz: int = 512,
                 format: str = "-",
                 device=None,
                 limit: int = None):
    """
    Benchmark para MODELOS DE SEGMENTAÇÃO (YOLO-Seg).
    Mede tempo de inferência, uso de recursos e métricas de MÁSCARA.
    Retorna [ {coluna: valor} ] pronto para DataFrame/CSV.
    """
    status = "✅"
    fmt_label = _format_label(format)
    size_mb = np.nan

    try:
        # 1) Carrega pesos
        base = YOLO(model)

        # 2) Exporta se necessário (ONNX/TRT) e reabre para inferência
        infer_model, export_artifact = _export_if_needed(base, format, device=device)
        size_mb = _model_size_mb(export_artifact if export_artifact else model)

        # 3) Carrega caminhos de validação
        val_paths = _load_val_image_paths(data)
        if limit is not None and isinstance(limit, int) and limit > 0:
            val_paths = val_paths[:limit]

        # 4) Loop de inferência
        inf_times = []
        cpu_usages, ram_usages = [], []
        gpu_usages, vram_usages, power_usages = [], [], []
        vram_proc_usages = []

        total_start = time.time()
        for img_path in val_paths:
            cpu_b, ram_b, gpu_b, vram_b, pwr_b, vram_proc_b = get_system_usage()
            t0 = time.time()

            _ = infer_model.predict(
                source=img_path,
                imgsz=imgsz,
                device=device,
                verbose=False
            )

            dt = (time.time() - t0)
            cpu_a, ram_a, gpu_a, vram_a, pwr_a, vram_proc_a = get_system_usage()

            inf_times.append(dt)
            cpu_usages.append((cpu_b + cpu_a) / 2.0)
            ram_usages.append((ram_b + ram_a) / 2.0)
            gpu_usages.append((gpu_b + gpu_a) / 2.0)
            vram_usages.append((vram_b + vram_a) / 2.0)
            power_usages.append((pwr_b + pwr_a) / 2.0)
            vram_proc_usages.append((vram_proc_b + vram_proc_a) / 2.0)

        total_time = max(1e-9, time.time() - total_start)
        n_imgs = len(inf_times)

        mean_ms = (np.mean(inf_times) * 1000.0) if n_imgs else np.nan
        fps = (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else 0.0
        throughput = (n_imgs / total_time) if total_time > 0 else 0.0

        # 5) Métricas de SEGMENTAÇÃO do val()
        metrics = _safe_val_metrics_seg(infer_model, data=data, imgsz=imgsz, device=device)

        def _r(x, nd=4):
            try:
                v = float(x)
                if math.isnan(v):
                    return np.nan
                return round(v, nd)
            except Exception:
                return np.nan

        row = {
            "Format": fmt_label,
            "Status❔": status,
            "Size (MB)": size_mb,

            # --- métricas de MÁSCARA ---
            "metrics/seg/mAP50-95": _r(metrics.get("metrics/seg/mAP50-95"), 4),
            "metrics/seg/mAP50": _r(metrics.get("metrics/seg/mAP50"), 4),
            "metrics/seg/precision": _r(metrics.get("metrics/seg/precision"), 4),
            "metrics/seg/recall": _r(metrics.get("metrics/seg/recall"), 4),
            "metrics/seg/f1": _r(metrics.get("metrics/seg/f1"), 4),
            "metrics/seg/accuracy": _r(metrics.get("metrics/seg/accuracy"), 4),

            # --- desempenho ---
            "Inference time (ms/im)": round(mean_ms, 2) if mean_ms == mean_ms else np.nan,
            "FPS": round(fps, 2),
            "Throughput (img/s)": round(throughput, 2),

            # --- recursos (por processo quando possível) ---
            "cpu_proc_avg (%)": round(float(np.mean(cpu_usages)) if cpu_usages else np.nan, 2),
            "ram_proc_mb (MB)": round(float(np.mean(ram_usages)) if ram_usages else np.nan, 1),
            "gpu_avg (%)": round(float(np.mean(gpu_usages)) if gpu_usages else np.nan, 2),           # global
            "vram_avg_mb (MB)": round(float(np.mean(vram_usages)) if vram_usages else np.nan, 1),   # global
            "vram_proc_mb (MB)": round(float(np.mean(vram_proc_usages)) if vram_proc_usages else np.nan, 1),
            "power_avg (W)": round(float(np.mean(power_usages)) if power_usages else np.nan, 2),
        }

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
