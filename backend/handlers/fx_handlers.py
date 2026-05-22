"""FX rack and insert control message handlers."""


def register_handlers(server):
    async def block_quarantined_write(websocket, surface, result_type, request):
        blocker = getattr(server, "_block_quarantined_wing_write_surface", None)
        if blocker is None:
            return False
        return await blocker(
            websocket,
            surface=surface,
            result_type=result_type,
            request=request,
        )

    async def handle_get_fx_overview(websocket, data):
        if not server.mixer_client:
            return
        fx_slots = []
        for slot in range(1, 17):
            slot_name = f"FX{slot}"
            if hasattr(server.mixer_client, "get_fx_slot_info"):
                fx_slots.append(server.mixer_client.get_fx_slot_info(slot_name))
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "fx_overview",
            "fx_slots": fx_slots,
        })

    async def handle_get_insert_state(websocket, data):
        if not server.mixer_client or not hasattr(server.mixer_client, "get_insert"):
            return
        target_type = data.get("target_type", "channel")
        target = data.get("target")
        position = data.get("position", "pre")
        insert_state = server.mixer_client.get_insert(target_type, target, position)
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "insert_state",
            "insert": insert_state,
        })

    async def handle_set_insert(websocket, data):
        if await block_quarantined_write(
            websocket,
            "set_insert",
            "insert_updated",
            {
                "target_type": data.get("target_type", "channel"),
                "target": data.get("target"),
                "position": data.get("position", "pre"),
                "slot": data.get("slot", "NONE"),
            },
        ):
            return
        if not server.mixer_client or not hasattr(server.mixer_client, "set_insert"):
            return
        server.mixer_client.set_insert(
            data.get("target_type", "channel"),
            data.get("target"),
            data.get("position", "pre"),
            slot=data.get("slot", "NONE"),
            on=data.get("on"),
            mode=data.get("mode"),
        )
        if hasattr(server.mixer_client, "get_insert"):
            insert_state = server.mixer_client.get_insert(
                data.get("target_type", "channel"),
                data.get("target"),
                data.get("position", "pre"),
            )
        else:
            insert_state = None
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "insert_updated",
            "insert": insert_state,
        })

    async def handle_set_fx_model(websocket, data):
        if await block_quarantined_write(
            websocket,
            "set_fx_model",
            "fx_slot_updated",
            {"slot": data.get("slot"), "model": data.get("model")},
        ):
            return
        if not server.mixer_client or not hasattr(server.mixer_client, "set_fx_model"):
            return
        slot = data.get("slot")
        server.mixer_client.set_fx_model(slot, data.get("model"))
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "fx_slot_updated",
            "fx": server.mixer_client.get_fx_slot_info(slot) if hasattr(server.mixer_client, "get_fx_slot_info") else None,
        })

    async def handle_set_fx_on(websocket, data):
        if await block_quarantined_write(
            websocket,
            "set_fx_on",
            "fx_slot_updated",
            {"slot": data.get("slot"), "on": data.get("on")},
        ):
            return
        if not server.mixer_client or not hasattr(server.mixer_client, "set_fx_on"):
            return
        slot = data.get("slot")
        server.mixer_client.set_fx_on(slot, data.get("on"))
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "fx_slot_updated",
            "fx": server.mixer_client.get_fx_slot_info(slot) if hasattr(server.mixer_client, "get_fx_slot_info") else None,
        })

    async def handle_set_fx_mix(websocket, data):
        if await block_quarantined_write(
            websocket,
            "set_fx_mix",
            "fx_slot_updated",
            {"slot": data.get("slot"), "mix": data.get("mix")},
        ):
            return
        if not server.mixer_client or not hasattr(server.mixer_client, "set_fx_mix"):
            return
        slot = data.get("slot")
        server.mixer_client.set_fx_mix(slot, data.get("mix"))
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "fx_slot_updated",
            "fx": server.mixer_client.get_fx_slot_info(slot) if hasattr(server.mixer_client, "get_fx_slot_info") else None,
        })

    async def handle_set_fx_parameter(websocket, data):
        if await block_quarantined_write(
            websocket,
            "set_fx_parameter",
            "fx_parameter_updated",
            {
                "slot": data.get("slot"),
                "parameter": data.get("parameter"),
                "value": data.get("value"),
            },
        ):
            return
        if not server.mixer_client or not hasattr(server.mixer_client, "set_fx_parameter"):
            return
        slot = data.get("slot")
        parameter = data.get("parameter")
        value = data.get("value")
        server.mixer_client.set_fx_parameter(slot, parameter, value)
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "fx_parameter_updated",
            "slot": slot,
            "parameter": parameter,
            "value": value,
        })

    return {
        "get_fx_overview": handle_get_fx_overview,
        "get_insert_state": handle_get_insert_state,
        "set_insert": handle_set_insert,
        "set_fx_model": handle_set_fx_model,
        "set_fx_on": handle_set_fx_on,
        "set_fx_mix": handle_set_fx_mix,
        "set_fx_parameter": handle_set_fx_parameter,
    }
