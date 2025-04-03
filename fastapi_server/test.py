@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    """
    logger.info("New WebSocket connection attempt")

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        active_connections.add(websocket)
        logger.info(f"Active WebSocket connections: {len(active_connections)}")

        try:
            await process_frame(websocket)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection: {str(e)}", exc_info=True)
        
    finally:
        # Remove from active connections
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info("WebSocket removed from active connections")

        # Close the WebSocket connection
        await websocket.close()
        logger.info("WebSocket connection closed")

    return {"status": "running", "active_connections": len(active_connections)}