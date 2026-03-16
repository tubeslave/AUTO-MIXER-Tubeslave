"""Output and channel input routing message handlers."""

from wing_client import WingClient
from osc.enhanced_osc_client import EnhancedOSCClient


def register_handlers(server):
    async def handle_route_output(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            output_group = data.get("output_group")
            output_number = data.get("output_number")
            source_group = data.get("source_group")
            source_channel = data.get("source_channel")

            if all([output_group, output_number is not None, source_group, source_channel is not None]):
                success = server.mixer_client.route_output(
                    output_group, output_number, source_group, source_channel
                )
                await server.send_to_client(websocket, {
                    "type": "route_output_result",
                    "success": success,
                    "output_group": output_group,
                    "output_number": output_number,
                    "source_group": source_group,
                    "source_channel": source_channel
                })

    async def handle_route_multiple_outputs(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            output_group = data.get("output_group")
            start_output = data.get("start_output")
            num_outputs = data.get("num_outputs")
            source_group = data.get("source_group")
            start_source_channel = data.get("start_source_channel")

            if all([output_group, start_output is not None, num_outputs is not None,
                    source_group, start_source_channel is not None]):
                success_count = server.mixer_client.route_multiple_outputs(
                    output_group, start_output, num_outputs, source_group, start_source_channel
                )
                await server.send_to_client(websocket, {
                    "type": "route_multiple_outputs_result",
                    "success_count": success_count,
                    "total": num_outputs,
                    "output_group": output_group,
                    "start_output": start_output,
                    "source_group": source_group,
                    "start_source_channel": start_source_channel
                })

    async def handle_get_output_routing(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            output_group = data.get("output_group")
            output_number = data.get("output_number")

            if output_group and output_number is not None:
                routing = server.mixer_client.get_output_routing(output_group, output_number)
                await server.send_to_client(websocket, {
                    "type": "output_routing",
                    "routing": routing
                })

    async def handle_set_channel_input(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            channel = data.get("channel")
            source_group = data.get("source_group")
            source_channel = data.get("source_channel")

            if all([channel is not None, source_group, source_channel is not None]):
                success = server.mixer_client.set_channel_input(channel, source_group, source_channel)
                await server.send_to_client(websocket, {
                    "type": "set_channel_input_result",
                    "success": success,
                    "channel": channel,
                    "source_group": source_group,
                    "source_channel": source_channel
                })

    async def handle_set_channel_alt_input(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            channel = data.get("channel")
            source_group = data.get("source_group")
            source_channel = data.get("source_channel")

            if all([channel is not None, source_group, source_channel is not None]):
                success = server.mixer_client.set_channel_alt_input(channel, source_group, source_channel)
                await server.send_to_client(websocket, {
                    "type": "set_channel_alt_input_result",
                    "success": success,
                    "channel": channel,
                    "source_group": source_group,
                    "source_channel": source_channel
                })

    async def handle_get_channel_input_routing(websocket, data):
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            channel = data.get("channel")

            if channel is not None:
                routing = server.mixer_client.get_channel_input_routing(channel)
                await server.send_to_client(websocket, {
                    "type": "channel_input_routing",
                    "routing": routing
                })

    return {
        "route_output": handle_route_output,
        "route_multiple_outputs": handle_route_multiple_outputs,
        "get_output_routing": handle_get_output_routing,
        "set_channel_input": handle_set_channel_input,
        "set_channel_alt_input": handle_set_channel_alt_input,
        "get_channel_input_routing": handle_get_channel_input_routing,
    }
