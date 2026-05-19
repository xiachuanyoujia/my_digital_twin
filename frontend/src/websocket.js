export class PoseWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.onPoseFrame = null;
    this.onStatusChange = null;
    this.reconnectDelay = 1500;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this._updateStatus(true);
      console.log("[WS] Connected to backend");
    };

    this.ws.onclose = () => {
      this._updateStatus(false);
      console.log("[WS] Disconnected, reconnecting...");
      setTimeout(() => this.connect(), this.reconnectDelay);
    };

    this.ws.onerror = (e) => {
      console.error("[WS] Error:", e);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.onPoseFrame) {
        this.onPoseFrame(data);
      }
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
      el.textContent = "Backend: Connected";
      el.className = "connected";
    } else {
      el.textContent = "Backend: Disconnected";
      el.className = "disconnected";
    }
    if (this.onStatusChange) {
      this.onStatusChange(connected);
    }
  }
}
