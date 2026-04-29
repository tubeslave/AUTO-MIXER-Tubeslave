"""Training service handlers for automatic ML refresh via internet datasets."""



def register_handlers(server):

    async def _get_service():
        return getattr(server, "agent_training_service", None)

    async def handle_get_training_status(websocket, data):
        service = await _get_service()
        if service is None:
            await server.send_to_client(websocket, {
                "type": "training_status",
                "enabled": False,
                "running": False,
                "error": "training service unavailable",
            })
            return

        status = service.get_status()
        await server.send_to_client(websocket, {
            "type": "training_status",
            "status": status,
        })

    async def handle_start_training(websocket, data):
        service = await _get_service()
        if service is None:
            await server.send_to_client(websocket, {
                "type": "training_start_failed",
                "error": "training service unavailable",
            })
            return

        force = bool(data.get("force", False))
        reason = str(data.get("reason", "manual"))
        manifest_url = data.get("manifest_url")
        if not isinstance(manifest_url, str) or not manifest_url.strip():
            manifest_url = None

        result = await service.start_once(force=force, reason=reason, manifest_url=manifest_url)
        await server.send_to_client(websocket, {
            "type": "training_started",
            "result": result,
            "training_status": service.get_status(),
        })

    async def handle_stop_training(websocket, data):
        service = await _get_service()
        if service is None:
            await server.send_to_client(websocket, {
                "type": "training_stopped",
                "error": "training service unavailable",
            })
            return

        await service.stop()
        await server.send_to_client(websocket, {
            "type": "training_stopped",
            "training_status": service.get_status(),
        })

    return {
        "get_training_status": handle_get_training_status,
        "start_training": handle_start_training,
        "stop_training": handle_stop_training,
    }
