const BACKEND_LABELS = {
  mediapipe: "MediaPipe",
  yolo: "YOLO",
  hybrid: "混合模式",
};

export class PoseWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.onPoseFrame = null;
    this.onStatusChange = null;
    this.reconnectDelay = 1500;
    this.maxReconnectDelay = 15000;
    this._lastBackend = "";
    this._currentDelay = 1500;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this._currentDelay = this.reconnectDelay;
      if (this._noDataTimer) { clearTimeout(this._noDataTimer); this._noDataTimer = null; }
      this._updateStatus(true);
      console.log("[WS] Connected to backend");
    };

    this.ws.onclose = () => {
      this._updateStatus(false);
      console.log(`[WS] Disconnected, reconnecting in ${this._currentDelay}ms...`);
      setTimeout(() => this.connect(), this._currentDelay);
      this._currentDelay = Math.min(this._currentDelay * 1.5, this.maxReconnectDelay);
    };

    this.ws.onerror = (e) => {
      console.error("[WS] Error:", e);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // 系统消息 (连接确认等)
      if (data.type === "connected") {
        console.log(`[WS] 收到确认: backend=${data.backend}`);
        this._msgCount = 0;
        this._lastLogTime = Date.now();
        if (data.backend) {
          this._lastBackend = data.backend;
          this._updateBackendLabel(data.backend);
        }
        // 启动"无数据"检测定时器
        this._startNoDataCheck();
        return;
      }
      // 姿态数据
      this._msgCount = (this._msgCount || 0) + 1;
      if (this._msgCount <= 5 || this._msgCount % 30 === 0) {
        const now = Date.now();
        const elapsed = this._lastLogTime ? ((now - this._lastLogTime) / 1000).toFixed(1) : "?";
        console.log(`[WS] msg #${this._msgCount} type=${data.type || "pose"} backend=${data.backend || "?"} ts=${data.timestamp?.toFixed?.(1) || data.timestamp} +${elapsed}s`);
        this._lastLogTime = now;
      }
      if (data.backend && this._lastBackend !== data.backend) {
        this._lastBackend = data.backend;
        this._updateBackendLabel(data.backend);
      }
      if (this.onPoseFrame) {
        this.onPoseFrame(data);
      }
    };

    this._startNoDataCheck = () => {
      this._noDataTimer = setTimeout(() => {
        if ((this._msgCount || 0) === 0) {
          console.warn("[WS] 已连接但 3 秒内未收到姿态数据! 检查后端摄像头是否正常");
        }
      }, 3000);
    };
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  _updateStatus(connected) {
    const el = document.getElementById("status");
    if (!el) return;
    if (connected) {
      const label = BACKEND_LABELS[this._lastBackend] || this._lastBackend;
      el.textContent = `后端: 已连接${label ? ` (${label})` : ""}`;
      el.className = "connected";
    } else {
      el.textContent = "后端: 已断开";
      el.className = "disconnected";
    }
    if (this.onStatusChange) {
      this.onStatusChange(connected);
    }
  }

  _updateBackendLabel(backend) {
    const el = document.getElementById("status");
    if (!el || el.className !== "connected") return;
    const label = BACKEND_LABELS[backend] || backend;
    el.textContent = `后端: 已连接 (${label})`;
  }
}
