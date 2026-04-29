# iPhone Remote Setup

This project now includes a touch-first remote control surface for iPhone/iPad. It is
served by the MacBook frontend and installed on iPhone as a Safari Home Screen web app.

## What You Need

- MacBook and iPhone/iPad on the same Wi-Fi network.
- Backend running on the MacBook and reachable on port `8765`.
- Frontend running on the MacBook and reachable on port `3000`.
- Mixer reachable from the MacBook network.

## Start on MacBook

From the repository root:

```bash
cd backend
python server.py
```

In another terminal:

```bash
cd frontend
npm install
npm start
```

The backend binds to `0.0.0.0:8765` when started with `python server.py`, so iPhone can
reach it over the local network. The frontend Vite server already listens on
`0.0.0.0:3000`.

Find the MacBook Wi-Fi IP:

```bash
ipconfig getifaddr en0
```

Example result: `192.168.1.20`.

## Open and Install on iPhone

1. Connect iPhone to the same Wi-Fi as the MacBook.
2. Open Safari on iPhone.
3. Go to `http://<MACBOOK_IP>:3000`, for example:
   `http://192.168.1.20:3000`.
4. Safari opens the mobile `Remote` screen automatically.
5. Tap Share, then Add to Home Screen.
6. Name it `AUTO MIXER` and tap Add.

After that, launch AUTO MIXER from the iPhone Home Screen.

## Configure Connection in the Remote

Open the `Setup` tab in the bottom navigation.

1. Backend on MacBook:
   - The app automatically uses `ws://<MACBOOK_IP>:8765` when opened from
     `http://<MACBOOK_IP>:3000`.
   - If needed, edit the Backend field manually and tap OK.
2. Mixer:
   - Choose `Behringer WING`, `Allen & Heath dLive`, or `Mixing Station`.
   - Enter mixer IP/host and port.
   - Tap Connect.
3. Audio interface:
   - Choose Dante Virtual Soundcard, SoundGrid, WING USB, or another available input.
   - Tap Scan audio if the list needs refresh.
4. Channels:
   - Use Channels to enable the sources AUTO-MIXER may process.
   - Use channel-name scan after the mixer is connected.
5. Safety:
   - Tap Backup before applying corrections.
   - Use Apply for pending AI actions.
   - Use Otkat to restore the last channel backup.
   - Use STOP for AI emergency stop.
   - Use Bypass only intentionally: on WING it disables processing and resets channel
     faders to `0 dB`.

## Network Troubleshooting

- If iPhone opens the page but shows `Offline`, check that `python server.py` is still
  running and that macOS Firewall allows Python on port `8765`.
- If the page does not open, check that `npm start` is running and macOS Firewall allows
  Node/Vite on port `3000`.
- If the mixer does not connect, test the same mixer IP from the MacBook first.
- If the MacBook IP changes after reconnecting Wi-Fi, reopen Safari with the new IP and
  update the Home Screen shortcut if needed.

## Optional Environment Overrides

```bash
AUTOMIXER_WS_HOST=0.0.0.0 AUTOMIXER_WS_PORT=8765 python backend/server.py
```

Use a narrower host only when you intentionally do not want iPhone/iPad access.
