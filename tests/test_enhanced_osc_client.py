from unittest.mock import MagicMock, patch

from osc.enhanced_osc_client import ConnectionState, EnhancedOSCClient
from wing_client import verify_wing_physical_readback_receipt


@patch("wing_client.WingClient")
def test_transport_generation_changes_after_internal_reconnect(wing_client):
    wing = MagicMock()
    wing.connect.return_value = True
    wing.is_connected = True
    wing_client.return_value = wing
    client = EnhancedOSCClient(reconnect_interval=0.0, max_reconnect_attempts=1)

    assert client.transport_generation == 0
    assert client.connect() is True
    assert client.transport_generation == 1

    client._stop_reconnect = False
    client._reconnect_loop()

    assert client.transport_generation == 2
    assert client.get_stats()["reconnect_count"] == 1


@patch("wing_client.WingClient")
def test_disconnect_invalidates_generation_before_reconnect_connect_returns(
    wing_client,
):
    wing = MagicMock()
    wing.connect.return_value = True
    wing.is_connected = True
    wing_client.return_value = wing
    client = EnhancedOSCClient()
    assert client.connect() is True

    with patch.object(client, "_start_reconnect_loop") as start_reconnect:
        client._handle_disconnect()

    assert client.transport_generation == 2
    assert client.connection_state is ConnectionState.RECONNECTING
    assert wing.is_connected is True
    start_reconnect.assert_called_once_with()


@patch("wing_client.WingClient")
def test_query_parameter_routes_send_failure_through_reconnect_boundary(
    wing_client,
):
    wing = MagicMock()
    wing.connect.return_value = True
    wing.is_connected = True
    wing_client.return_value = wing
    client = EnhancedOSCClient()
    assert client.connect() is True

    def query(address, timeout, *, send_callable):
        wing.is_connected = False
        assert send_callable(address) is False
        return {"status": "send_failed", "address": address}

    wing.query_parameter.side_effect = query
    wing.send.return_value = False
    with patch.object(client, "_start_reconnect_loop"):
        result = client.query_parameter("/$syscfg/$serial", 0.1)

    assert result["status"] == "send_failed"
    assert client.connection_state is ConnectionState.RECONNECTING
    assert client.transport_generation == 2


@patch("wing_client.WingClient")
def test_explicit_disconnect_cannot_be_reopened_by_stale_send_failure(
    wing_client,
):
    wing = MagicMock()
    wing.connect.return_value = True
    wing.is_connected = True
    wing_client.return_value = wing
    client = EnhancedOSCClient()
    assert client.connect() is True
    client.disconnect()
    wing.send.return_value = False
    wing.is_connected = False

    with patch.object(client, "_start_reconnect_loop") as start_reconnect:
        assert client.send("/$syscfg/$serial") is False

    assert client.connection_state is ConnectionState.DISCONNECTED
    start_reconnect.assert_not_called()


@patch("wing_client.WingClient")
def test_mutation_boundary_match_check_delegates_to_wing_client(wing_client):
    wing = MagicMock()
    wing.mutation_boundary_matches.return_value = True
    wing_client.return_value = wing
    client = EnhancedOSCClient()
    token = object()

    assert client.mutation_boundary_matches(token) is True
    wing.mutation_boundary_matches.assert_called_once_with(token)


def test_canonical_wrapper_receipt_is_bound_and_single_use():
    client = EnhancedOSCClient(ip="127.0.0.1", port=2223)
    client._wing.is_connected = True
    client._wing.sock = object()
    replies = iter((-12.1, -12.0))

    def send(address, *values):
        client._wing._handle_transport_message(
            address,
            (next(replies),),
            ("127.0.0.1", 2223),
        )
        return True

    client._wing.send = send
    receipt = client.query_parameter("/ch/3/fdr", 0.01)

    assert verify_wing_physical_readback_receipt(
        client,
        receipt,
        "/ch/3/fdr",
    ) is False
    assert verify_wing_physical_readback_receipt(
        client,
        receipt,
        "/ch/3/fdr",
    ) is False
