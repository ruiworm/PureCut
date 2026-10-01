/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./scripts/studio_template.html'],
  theme: {
    extend: {
      colors: {
        app: {
          bg: '#121214',
          surface: '#18191c',
          card: '#1e1f23',
          cardHover: '#25262c',
          border: '#2a2b32',
          borderLight: '#363842',
          emerald: '#10b981',
          emeraldHover: '#059669',
          emeraldBg: 'rgba(16, 185, 129, 0.12)',
          amber: '#f59e0b',
          rose: '#f43f5e',
          text: '#e5e7eb',
          textMuted: '#8b8e9b',
          textSubtle: '#5c5f6d'
        }
      }
    }
  }
};
