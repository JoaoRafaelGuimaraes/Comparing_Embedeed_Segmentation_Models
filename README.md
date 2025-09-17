# Benchmark de Modelos de Segmentação em Hardwares Embarcados  

Este projeto permite avaliar o desempenho de **modelos de segmentação** (YOLO e DeepLab) em diferentes hardwares embarcados, como **NVIDIA Jetsons** e **Raspberry Pi**.  
Os benchmarks medem métricas de **tempo de inferência, precisão, recall e F1-score**, exportando os resultados em formato **CSV** para comparação.  

---

## Como clonar 
Para ter acesso às imagens e modelos, é necessário clonar o repositório com o seguinte comando:

```bash
git clone --recursive https://github.com/JoaoRafaelGuimaraes/Comparing_Embedeed_Segmentation_Models.git
```

## 📌 Estrutura do Projeto  

```
├── benchmark.py          # Script principal de benchmark
├── deep_lab.py           #Classes para realizar inferencia com modelo DeepLab
├── gpu_monitor.py         #Classe para monitorar a utilização de GPU a partir do comando "tegrastats", nativo da jetson
├── benchmark_PI.py       # Funções auxiliares para benchmarking em Jetson/Raspberry
├── models/               # Modelos (.pt, .onnx, .engine, etc.)
├── dataset/              # Dataset em formato YOLO/segmentation
├── bench_results.csv     # Resultados principais
└── README.md
```

---

## ⚡ Funcionalidades  

- Avaliação de **diferentes formatos** de modelos:
  - **CPU** → ONNX, PyTorch , DeepLab  
  - **GPU** → TensorRT (`engineFP32`, `engineFP16`), ONNX, PyTorch (`-`), DeepLab  
- Resultados exportados automaticamente em CSV.  


---

## ▶️ Como rodar o benchmark  

O script `benchmark.py` precisa receber o **modo de execução**: `CPU` ou `GPU` e o nome do dispositivo!!! 


### Exemplo com JetsonOrin
```bash
python3 benchmark.py GPU -d JetsonOrin16Gb
```

### Exemplo com Raspberry Pi (CPU only)  
```bash
python3 benchmark.py CPU -d raspberrypi4
```

---

## 📊 Saída  

- Resultados salvos em:
  - `bench_results.csv` → consolidado dos testes  

Cada linha contém:
- Formato do modelo  
- Métricas (precisão, recall, F1-score, tempo de inferência)  
- Dispositivo utilizado  

---
