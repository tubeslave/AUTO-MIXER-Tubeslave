/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#070b10',
        bg2: '#0b1118',
        panel: '#101820',
        panel2: '#151f29',
        row: '#0d141d',
        line: '#233141',
        line2: '#2f4053',
        text: '#e6edf3',
        muted: '#8b949e',
        dim: '#5f6b78',
        online: '#00d26a',
        ok: '#00d26a',
        warn: '#ff9f1a',
        danger: '#ff4d4f',
        audio: '#1687ff',
        osc: '#1687ff',
        ai: '#a855f7'
      },
      fontFamily: {
        ui: ['Inter', 'SF Pro Display', 'Segoe UI', 'Arial', 'sans-serif'],
        mono: ['SFMono-Regular', 'Menlo', 'Consolas', 'monospace']
      },
      boxShadow: {
        glow: '0 0 22px rgba(0, 212, 255, 0.16)',
        panel: 'inset 0 1px 0 rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.22)'
      }
    }
  },
  plugins: []
};
