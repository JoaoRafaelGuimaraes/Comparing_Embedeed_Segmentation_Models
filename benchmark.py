from ultralytics.utils.benchmarks import benchmark
import pandas as pd
import sys
import os
# Precision, Recall e F1 score

if (len(sys.argv) < 2):
    print('Por favor, complete os campos:\npython3 benchmark.py <device> (CPU ou GPU)')
    sys.exit(1)
arg = sys.argv[1]
device = None

if len(sys.argv)>2 and sys.argv[2] == '-d' and len(sys.argv)>3:
    device= sys.argv[3]


print(f'Rodando em uma {arg}, eu um {device}\n')
formats_CPU = ['onnx', '-'] #Na cpu, roda apenas onxx e PyTorch
formats_GPU = ['engine', 'onnx', '-'] #Na gpu, roda engine, onxx e PyTorch normal

my_formats = formats_CPU if arg=='CPU' else formats_GPU

csv_path = "bench_results.csv"

for format in my_formats:
    if arg == 'GPU':
        results = benchmark(model="models/land-seg.pt", data="dataset/landslide_dataset_1000/data.yml", imgsz=512, format=format, device =0)
    else: 
        results = benchmark(model="models/land-seg.pt", data="dataset/landslide_dataset_1000/data.yml", imgsz=512, format=format)
    df = pd.DataFrame(results)
    df['device'] = device

    if not os.path.exists(csv_path):
        print(f'Criando o arquivo CSV em {csv_path}')
        df.to_csv(csv_path, index=False, mode="w")
    else:
        print(f'Linhas adicionadas ao arquivo CSV salvo em {csv_path}')
        df.to_csv(csv_path, index=False,mode='a', header='False')
