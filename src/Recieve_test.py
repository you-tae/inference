import subprocess
import socket
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys

# --- ADB reverse 자동 실행 ---
def setup_adb_reverse():
    try:
        print("📡 Setting up ADB reverse...")
        subprocess.run(['adb', 'reverse', 'tcp:5005', 'tcp:5005'], check=True)
        print("✅ ADB reverse successful.")
    except subprocess.CalledProcessError as e:
        print("❗ ADB reverse failed:", e)
        sys.exit(1)

# --- 서버 설정 ---
TCP_IP = "0.0.0.0"
TCP_PORT = 5005
BUFFER_SIZE = 32768
SAMPLE_RATE = 24000
TIME_WINDOW = 1.0  # 초 단위

# 샘플 수 계산
max_samples = int(SAMPLE_RATE * TIME_WINDOW)
buf_l = np.zeros(max_samples, dtype=np.int16)
buf_r = np.zeros(max_samples, dtype=np.int16)

# --- pyqtgraph 창 구성 ---
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-time MIC Monitor (Time + Frequency Domain)")
win.show()
win.resize(1000, 800)

# --- Time Domain Plots ---
plot_l = win.addPlot(title="Left Channel - Time Domain")
curve_l = plot_l.plot(pen='c')
plot_l.setYRange(-32000, 32000)

win.nextRow()
plot_r = win.addPlot(title="Right Channel - Time Domain")
curve_r = plot_r.plot(pen='m')
plot_r.setYRange(-32000, 32000)

# --- Frequency Domain Plots (Magnitude) ---
win.nextRow()
plot_fft_l = win.addPlot(title="Left Channel - Frequency Domain (Magnitude)")
curve_fft_l = plot_fft_l.plot(pen='c')
plot_fft_l.enableAutoRange(axis='y', enable=True)

win.nextRow()
plot_fft_r = win.addPlot(title="Right Channel - Frequency Domain (Magnitude)")
curve_fft_r = plot_fft_r.plot(pen='m')
plot_fft_r.enableAutoRange(axis='y', enable=True)

# 시간 / 주파수 축 계산
t_data = np.linspace(-TIME_WINDOW, 0, max_samples)
freq_data = np.fft.rfftfreq(max_samples, d=1 / SAMPLE_RATE)

# --- 서버 소켓 열기 ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen(1)

print(f"📌 Listening on {TCP_IP}:{TCP_PORT}...")
conn, addr = server_socket.accept()
print(f"✅ Connected to {addr}")

# --- 업데이트 함수 ---
def update():
    global buf_l, buf_r
    try:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            print("❗ Connection closed")
            sys.exit(0)

        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) % 2 != 0:
            samples = samples[:-1]

        left = samples[::2]
        right = samples[1::2]

        buf_l = np.concatenate((buf_l, left))[-max_samples:]
        buf_r = np.concatenate((buf_r, right))[-max_samples:]

        # --- 시간 도메인 ---
        curve_l.setData(t_data, buf_l)
        curve_r.setData(t_data, buf_r)

        # --- 주파수 도메인 (Magnitude) ---
        fft_l = np.abs(np.fft.rfft(buf_l))
        fft_r = np.abs(np.fft.rfft(buf_r))

        curve_fft_l.setData(freq_data, fft_l)
        curve_fft_r.setData(freq_data, fft_r)

    except Exception as e:
        print("⚠️ Error during update:", e)
        sys.exit(1)

# --- 타이머 루프 ---
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)  # 30 FPS

# --- 실행 ---
if __name__ == '__main__':
    setup_adb_reverse()
    QtWidgets.QApplication.instance().exec_()
    conn.close()
    server_socket.close()
