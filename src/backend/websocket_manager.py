import json
import asyncio
from typing import List
from fastapi import WebSocket

class WebSocketConnectionManager:
    """Manages active WebSocket client connections for real-time telemetry and diagnostic trace streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                to_remove.append(connection)
                
        for conn in to_remove:
            self.disconnect(conn)

ws_manager = WebSocketConnectionManager()
