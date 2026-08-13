/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1a1a18",
        paper: "#faf9f6",
        line: "#e4e1d8",
        accent: "#2f5d50",
        accentSoft: "#e7efeb",
        warn: "#8a5a1d",
        warnSoft: "#f6ecdd",
      },
    },
  },
  plugins: [],
};
