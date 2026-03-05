from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO
import pandas as pd
import sys
import os
import platform
import shutil
import subprocess
from benchmark_PI import benchmark_PI


if (len(sys.argv) < 2):
    print('Por favor, complete os campos:\npython3 benchmark.py <device> (CPU ou GPU)')
    sys.exit(1)
arg = sys.argv[1]
device = None

if len(sys.argv)>2 and sys.argv[2] == '-d' and len(sys.argv)>3:
    device= sys.argv[3]


print(f'Rodando em uma {arg}, eu um {device}\n')
formats_CPU = ['onnx', '-', 'deeplab'] #Na cpu, roda apenas onxx e PyTorch
formats_GPU = ['engineFP32','engineFP16', 'onnx', '-', 'deeplab'] #Na gpu, roda engine, onxx e PyTorch normal


# is_jetson = shutil.which("tegrastats") is not None
# if not is_jetson:
#     formats_GPU = ['onnx', '-', 'deeplab'] # DESCOMENTE CASO DÊ ERRO EM EXPORTAR O MODELO DEVIDO AO TORCH!!!

my_formats = formats_CPU if arg=='CPU' else formats_GPU

csv_path = "bench_results.csv"
csv_path_ultralytics = "bench_results_ultralytics.csv"
for format in my_formats:
    if arg == 'GPU':
        results = benchmark_PI(model="models/land-seg.pt", data="dataset/landslide_dataset_1000/data.yml", imgsz=512, format=format, device =0)
    else: 
        results = benchmark_PI(model="models/land-seg.pt", data="dataset/landslide_dataset_1000/data.yml", imgsz=512, format=format)
    df = pd.DataFrame(results)
    df_ultr = pd.DataFrame(results)

    if isinstance(results, (list, tuple)) and len(results) >= 2 and isinstance(results[0], (list, tuple)):
        df = pd.DataFrame(results[1:], columns=results[0])
    else:
        df = pd.DataFrame(results)

    df['device'] = device

    if not os.path.exists(csv_path):
        print(f'Criando o arquivo CSV em {csv_path}')
        df.to_csv(csv_path, index=False, mode="w")
    else:
        print(f'Linhas adicionadas ao arquivo CSV salvo em {csv_path}')
        df.to_csv(csv_path, index=False,mode='a', header=False)


# ======================================================================
# Informações de versão (bibliotecas, JetPack, CUDA, TensorRT)
# ======================================================================
def print_version_info():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  INFORMAÇÕES DE VERSÃO DO AMBIENTE")
    print(sep)

    # Python
    print(f"Python          : {platform.python_version()}")
    print(f"Plataforma      : {platform.platform()}")
    print(f"Arquitetura     : {platform.machine()}")

    # JetPack / L4T
    is_jetson = shutil.which("tegrastats") is not None
    if is_jetson:
        # Tenta obter a versão do L4T (que mapeia para JetPack)
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                l4t_line = f.readline().strip()
            print(f"L4T (Tegra)     : {l4t_line}")
        except FileNotFoundError:
            pass
        # Tenta dpkg para versão exata do JetPack
        try:
            jp = subprocess.check_output(
                ["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            print(f"JetPack         : {jp}")
        except Exception:
            try:
                l4t_core = subprocess.check_output(
                    ["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                print(f"L4T Core        : {l4t_core}")
            except Exception:
                print("JetPack         : não encontrado via dpkg")
    else:
        print("JetPack         : N/A (não é Jetson)")

    # CUDA
    try:
        import torch
        print(f"PyTorch         : {torch.__version__}")
        print(f"CUDA (PyTorch)  : {torch.version.cuda or 'N/A'}")
        print(f"cuDNN           : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"CUDA disponível : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU             : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch         : não instalado")

    # TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT        : {trt.__version__}")
    except ImportError:
        print("TensorRT        : não instalado")

    # torchvision
    try:
        import torchvision
        print(f"torchvision     : {torchvision.__version__}")
    except ImportError:
        print("torchvision     : não instalado")

    # Ultralytics
    try:
        import ultralytics
        print(f"ultralytics     : {ultralytics.__version__}")
    except ImportError:
        print("ultralytics     : não instalado")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        ver = getattr(ort, '__version__', 'desconhecida')
        print(f"onnxruntime     : {ver}")
        if hasattr(ort, 'get_available_providers'):
            providers = ort.get_available_providers()
            print(f"ORT providers   : {', '.join(providers)}")
        else:
            print("ORT providers   : ⚠️  onnxruntime corrompido (sem get_available_providers)")
    except (ImportError, Exception) as e:
        print(f"onnxruntime     : não disponível ({e})")

    # OpenCV
    try:
        import cv2
        print(f"OpenCV          : {cv2.__version__}")
    except ImportError:
        print("OpenCV          : não instalado")

    # pandas / numpy
    print(f"pandas          : {pd.__version__}")
    try:
        import numpy as np
        print(f"numpy           : {np.__version__}")
    except ImportError:
        pass

    # psutil
    try:
        import psutil
        print(f"psutil          : {psutil.__version__}")
    except ImportError:
        pass

    # NVPModel e Jetson Clocks (modos de potência)
    is_jetson = shutil.which("tegrastats") is not None
    if is_jetson:
        try:
            nvp = subprocess.check_output(
                ["nvpmodel", "-q"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            # Primeira linha geralmente contém o modo (ex: "NV Power Mode: MAXN")
            for line in nvp.splitlines():
                if "power mode" in line.lower() or "mode" in line.lower():
                    print(f"NVPModel        : {line.strip()}")
                    break
            else:
                print(f"NVPModel        : {nvp.splitlines()[0].strip()}")
        except Exception:
            print("NVPModel        : não disponível")

        try:
            jc = subprocess.check_output(
                ["jetson_clocks", "--show"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            # Resumir: mostrar só as linhas de freq de GPU e CPU
            for line in jc.splitlines():
                low = line.lower()
                if "gpu" in low or "emc" in low:
                    print(f"Clocks          : {line.strip()}")
        except Exception:
            print("jetson_clocks   : não disponível (pode precisar sudo)")

    print(sep)


print_version_info()

