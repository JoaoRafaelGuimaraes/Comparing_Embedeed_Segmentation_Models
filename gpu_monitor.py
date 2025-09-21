# tegrastats_monitor.py
import subprocess, threading, re, numpy as np, shutil, time

class TegrastatsMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.stop_flag = False
        self.gpu = []
        self.power = []
        self.ram_used = []
        self.is_jetson = shutil.which("tegrastats") is not None

        # regex só usado em Jetson
        self._re_gpu = re.compile(r'GR3D_FREQ\s+(\d+)%')
        self._re_pwr = re.compile(r'VDD_IN\s+(\d+)mW(?:/\d+mW)?')
        self._re_ram = re.compile(r'RAM\s+(\d+)/(\d+)MB')

        if not self.is_jetson:
            import psutil
            self.psutil = psutil
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu_handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml = None
                self.gpu_handle = None

    def start(self):
        if self.is_jetson:
            self.proc = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1
            )
            self.thread = threading.Thread(target=self._reader_jetson, daemon=True)
        else:
            self.thread = threading.Thread(target=self._reader_pc, daemon=True)
        self.thread.start()

    def _reader_jetson(self):
        for line in self.proc.stdout:
            g = self._re_gpu.search(line)
            if g: self.gpu.append(float(g.group(1)))
            p = self._re_pwr.search(line)
            if p: self.power.append(float(p.group(1)) / 1000.0)
            r = self._re_ram.search(line)
            if r: self.ram_used.append(float(r.group(1)))
            if self.stop_flag: break

    def _reader_pc(self):
        while not self.stop_flag:
            # CPU RAM
            ram = self.psutil.virtual_memory().used / (1024**2)
            self.ram_used.append(ram)

            # GPU se disponível
            if self.nvml and self.gpu_handle:
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                power = self.nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                self.gpu.append(float(util.gpu))
                self.power.append(power)
            else:
                self.gpu.append(np.nan)
                self.power.append(np.nan)
                msg = ( "❌ GPU monitor indisponível!\n" "Verifique se os pacotes estão instalados no mesmo venv:\n" "    pip install psutil nvidia-ml-py3\n" )
                raise RuntimeError(msg)

            time.sleep(self.interval_ms / 1000.0)

    def stop_and_get(self):
        self.stop_flag = True
        try:
            if self.proc: self.proc.terminate()
        except Exception:
            pass
        if self.thread: self.thread.join(timeout=1.0)

        def mean(x): return float(np.mean(x)) if x else np.nan
        return mean(self.gpu), mean(self.power), mean(self.ram_used)
