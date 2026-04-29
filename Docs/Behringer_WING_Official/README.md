# Behringer WING Rack Official Docs

Official documents downloaded on 2026-04-17 for local use and protocol work.

## PDFs

- `pdfs/WING_Manual_WING_WING_RACK_WING_Compact.pdf`
- `pdfs/WING_RACK_QSG_All_Languages.pdf`
- `pdfs/WING_RACK_Dimensional_Drawing_2024-08-27.pdf`
- `pdfs/WING_Remote_Protocols_V3.1-03.pdf`
- `pdfs/WING_Remote_Protocols_FW_3.1_MediaValet.pdf`
- `pdfs/WING_Effects_Guide.pdf`
- `pdfs/WING_MIDI_Documentation.pdf`
- `pdfs/WING_DAW_Documentation.pdf`
- `pdfs/WING_DAW_Translation-Table.pdf`
- `pdfs/WING-DANTE_Update_Instructions_2025-07-23.pdf`
- `pdfs/AoIP_Dante_WSG_Module_Relocation_Instructions_2025-07-23.pdf`
- `pdfs/Waves_SoundGrid_QSG.pdf`

## Extracted Text

Each PDF has a matching `.txt` in `text/`, produced with `pdftotext` for fast local search.

Recommended commands:

```bash
rg -n "postins|preins|/fx/1/mdl|fxmix|MASTER|PLATE|PCORR" Docs/Behringer_WING_Official/text
rg -n "/main/1/preins|/mtx/1/preins|/bus/1/preins" Docs/Behringer_WING_Official/text/WING_Remote_Protocols_FW_3.1_MediaValet.txt
```

## Local Working Notes

- OSC + FX summary: `WING_RACK_OSC_FX_GUIDE.md`
- AI/backend knowledge mirror: `backend/ai/knowledge/wing_osc_reference.md`

## Most Important Source Files

- `WING_Remote_Protocols_FW_3.1_MediaValet.pdf`
  Main source for OSC addresses, node names, ranges, and model enums.
- `WING_Manual_WING_WING_RACK_WING_Compact.pdf`
  Main source for routing model, FX slot architecture, and workflow.
- `WING_Effects_Guide.pdf`
  Main source for FX module descriptions and practical usage notes.
