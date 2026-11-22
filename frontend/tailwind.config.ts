import type { Config } from "tailwindcss";

// Dark green primary with soft green highlights, on a very light neutral
// background. Kept deliberately narrow: no gradients, no bright accents.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#FAFAF8",
        surface: "#FFFFFF",
        ink: { DEFAULT: "#1A2420", muted: "#5B6B64", faint: "#8A9992" },
        leaf: {
          50: "#F2F7F4",
          100: "#E3EFE8",
          200: "#C7DFD1",
          300: "#9CC5AD",
          400: "#6BA384",
          500: "#44825F",
          600: "#2F6647",
          700: "#255239",
          800: "#1D412E",
          900: "#152F22",
        },
        warn: { 50: "#FDF6EC", 200: "#F2DCB8", 700: "#8A5A16" },
      },
      borderRadius: { xl: "0.875rem", "2xl": "1.25rem" },
      boxShadow: {
        card: "0 1px 2px rgba(21,47,34,0.04), 0 4px 16px rgba(21,47,34,0.05)",
        lift: "0 2px 4px rgba(21,47,34,0.06), 0 12px 32px rgba(21,47,34,0.08)",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        "fade-up": "fade-up 0.35s cubic-bezier(0.22,0.61,0.36,1) both",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
export default config;
