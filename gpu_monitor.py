# tegrastats_monitor.py
import subprocess, threading, re, numpy as np

class TegrastatsMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.stop_flag = False
        self.gpu = []
        self.power = []
        self.ram_used = []
        self._re_gpu = re.compile(r'GR3D_FREQ\s+(\d+)%')
        self._re_pwr = re.compile(r'POM_5V_GPU\s+(\d+)mW')
        self._re_ram = re.compile(r'RAM\s+(\d+)/(\d+)MB')

    def start(self):
        self.proc = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        for line in self.proc.stdout:
            g = self._re_gpu.search(line)
            if g: self.gpu.append(float(g.group(1)))
            p = self._re_pwr.search(line)
            if p: self.power.append(float(p.group(1)) / 1000.0)  # mW -> W
            r = self._re_ram.search(line)
            if r: self.ram_used.append(float(r.group(1)))        # MB usados
            if self.stop_flag:
                break

    def stop_and_get(self):
        self.stop_flag = True
        try:
            if self.proc: self.proc.terminate()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=1.0)

        def mean(x): return float(np.mean(x)) if x else 0.0
        return mean(self.gpu), mean(self.power), mean(self.ram_used)
