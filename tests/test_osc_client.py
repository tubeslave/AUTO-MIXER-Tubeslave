from integrations.mixing_station.models import MixingStationCommand
from integrations.mixing_station.osc_client import MixingStationOSCClient


def test_osc_client_builds_plain_value_address():
    client = MixingStationOSCClient(host="127.0.0.1", port=9000)
    command = MixingStationCommand(
        transport="osc",
        data_path="ch.0.mix.lvl",
        value=-5.0,
        value_format="plain",
    )

    message = client.build_message(command)

    assert message.address == "/con/v/ch.0.mix.lvl"
    assert message.value == -5.0


def test_osc_client_builds_normalized_value_address():
    client = MixingStationOSCClient()
    command = MixingStationCommand(
        transport="osc",
        data_path="ch.0.mix.pan",
        value=0.5,
        value_format="normalized",
    )

    message = client.build_message(command)

    assert message.address == "/con/n/ch.0.mix.pan"
