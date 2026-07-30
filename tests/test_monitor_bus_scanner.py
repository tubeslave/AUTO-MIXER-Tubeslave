"""Тесты классификатора мониторных шин (Ф10 этап 1, DEC-0010)."""

from monitor_bus_scanner import classify_bus_name, proposal_payload, scan_buses

MUSICIANS = ["Дима", "Вася", "Sergey"]


class TestClassify:
    def test_iem_english(self):
        c = classify_bus_name(1, "IEM Drums")
        assert c.kind == "iem" and c.confidence == 0.9

    def test_in_ear_variants(self):
        assert classify_bus_name(1, "In-Ear Vox").kind == "iem"
        assert classify_bus_name(2, "EARS Guitar").kind == "iem"
        assert classify_bus_name(3, "Уши барабаны").kind == "iem"

    def test_monitor_variants(self):
        assert classify_bus_name(1, "MON 3").kind == "monitor"
        assert classify_bus_name(2, "Monitor Vocal").kind == "monitor"
        assert classify_bus_name(3, "Монитор бас").kind == "monitor"

    def test_wedge(self):
        c = classify_bus_name(1, "Vocal Wedge")
        assert c.kind == "wedge" and c.confidence == 0.8

    def test_musician_name_match(self):
        c = classify_bus_name(5, "Дима IEM", MUSICIANS)
        assert c.kind == "iem" and c.musician == "Дима"
        assert c.confidence == 0.95

    def test_musician_only(self):
        c = classify_bus_name(5, "Вася", MUSICIANS)
        assert c.kind == "unknown" and c.musician == "Вася"
        assert c.confidence == 0.6 and c.is_monitoring

    def test_short_musician_alias_ignored(self):
        # короткие алиасы (<3 символов) не матчатся — ложные срабатывания
        c = classify_bus_name(1, "MONO SUM", ["Mo"])
        assert c.musician is None

    def test_ordinary_bus_is_unknown(self):
        c = classify_bus_name(7, "FX Reverb")
        assert not c.is_monitoring and c.confidence == 0.0

    def test_mon_does_not_match_inside_words(self):
        # \b-граница: Harmony/Ammonia не должны ловиться как MON
        assert classify_bus_name(1, "Harmony Bus").kind == "unknown"

    def test_empty_name(self):
        assert not classify_bus_name(1, "").is_monitoring


class TestScan:
    BUSES = {
        1: "IEM Дима",
        2: "MON 2",
        3: "FX Hall",
        4: "Vocal Wedge",
        5: "Drums subgroup",
        6: "Sergey ears",
    }

    def test_candidates_sorted_by_confidence(self):
        out = scan_buses(self.BUSES, MUSICIANS)
        assert [c.bus for c in out] == [1, 6, 4, 2]
        assert out[0].confidence == 0.95   # IEM + имя
        assert out[1].confidence == 0.95   # ears + имя

    def test_non_monitoring_excluded(self):
        out = scan_buses(self.BUSES, MUSICIANS)
        assert all(c.bus not in (3, 5) for c in out)

    def test_all_require_confirmation(self):
        assert all(c.requires_confirmation for c in scan_buses(self.BUSES, MUSICIANS))


class TestProposalPayload:
    def test_payload_shape_and_safety_flags(self):
        payload = proposal_payload(scan_buses(TestScan.BUSES, MUSICIANS))
        assert payload["type"] == "monitor_bus_mapping_proposal"
        assert payload["requires_confirmation"] is True
        assert payload["auto_routing"] is False
        first = payload["candidates"][0]
        assert set(first) == {"bus", "name", "kind", "musician", "confidence"}
