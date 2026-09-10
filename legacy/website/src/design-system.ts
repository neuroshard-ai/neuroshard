/*
 * ═══════════════════════════════════════════════════════════════
 *  NEUROSHARD DESIGN SYSTEM
 *  Quick reference for colors, typography, and patterns
 * ═══════════════════════════════════════════════════════════════
 *
 *
 *  BACKGROUNDS
 *  ───────────────────────────────────────────────────────────
 *  neutral-950   #0a0a0a    Page background, deepest level
 *  neutral-900   #171717    Cards, panels, containers
 *  neutral-800   #262626    Borders, dividers, subtle fills
 *  neutral-700   #404040    Hover borders, active elements
 *
 *
 *  TEXT
 *  ───────────────────────────────────────────────────────────
 *  white         #ffffff    Headings, primary emphasis
 *  neutral-200   #e5e5e5    Body text (default)
 *  neutral-400   #a3a3a3    Secondary text, descriptions
 *  neutral-500   #737373    Muted text, mono labels
 *  neutral-600   #525252    Disabled, fine print
 *
 *
 *  ACCENT
 *  ───────────────────────────────────────────────────────────
 *  accent        #c8ff00    Primary accent (lime-green)
 *  accent/90                Hover state for accent buttons
 *  accent/10                Subtle accent backgrounds
 *  accent/30                Accent borders
 *  accent/5                 Very faint accent tint
 *
 *
 *  SEMANTIC (status colors from Tailwind defaults)
 *  ───────────────────────────────────────────────────────────
 *  red-400       #f87171    Errors, destructive, sign out
 *  red-950/30               Error background tint
 *  green-400     #4ade80    Success, active/online
 *  green-500     #22c55e    Online indicator dots
 *  amber-400     #fbbf24    Warnings, pending states
 *  orange-400    #fb923c    Burn indicators
 *  purple-400    #c084fc    Proof types, special badges
 *
 *
 *  BUTTON PATTERNS
 *  ───────────────────────────────────────────────────────────
 *  Primary CTA:    bg-accent text-neutral-950 font-bold
 *  Secondary CTA:  bg-white text-neutral-950 font-bold
 *  Outline:        border border-neutral-700 text-white font-medium
 *  Subtle:         bg-neutral-800 text-neutral-300 font-medium
 *  Danger:         text-red-400 hover:bg-red-950/20
 *
 *  ⚠ ALWAYS use text-neutral-950 (black) on accent backgrounds
 *  ⚠ NEVER use text-white on bg-accent
 *
 *
 *  TYPOGRAPHY
 *  ───────────────────────────────────────────────────────────
 *  font-sans      "Instrument Sans"   Body text, UI elements
 *  font-display   "Space Grotesk"     Headings, display text
 *  font-mono      "JetBrains Mono"    Code, addresses, labels
 *
 *  Heading:       font-display text-4xl font-bold tracking-tight
 *  Section label: text-[10px] font-mono uppercase tracking-widest text-accent
 *  Mono label:    text-[10px] font-mono uppercase tracking-widest text-neutral-500
 *
 *
 *  SHAPES & EFFECTS
 *  ───────────────────────────────────────────────────────────
 *  ✗ No rounded corners   (no rounded-lg, rounded-xl, etc.)
 *  ✗ No gradients         (no bg-gradient-to-*)
 *  ✗ No glow shadows      (no shadow-[0_0_*px_*])
 *  ✗ No neon effects
 *  ✓ Sharp square edges on everything
 *  ✓ 1px borders          (border border-neutral-800)
 *  ✓ Minimal transitions  (transition-colors)
 *
 *
 *  LAYOUT PATTERNS
 *  ───────────────────────────────────────────────────────────
 *  Grid dividers:  Parent has gap-px bg-neutral-800,
 *                  children have bg-neutral-950
 *
 *  Max width:      max-w-7xl mx-auto px-6
 *
 *  Section gap:    py-24 for main sections
 *                  py-32 for hero-adjacent sections
 *
 *  Stat cards:     border border-neutral-800 bg-neutral-950/50 px-5 py-4
 *
 *
 * ═══════════════════════════════════════════════════════════════
 */

export const COLORS = {
  accent: '#c8ff00',
  bg: {
    page: '#0a0a0a',
    card: '#171717',
    border: '#262626',
    hoverBorder: '#404040',
  },
  text: {
    primary: '#ffffff',
    body: '#e5e5e5',
    secondary: '#a3a3a3',
    muted: '#737373',
    disabled: '#525252',
  },
} as const;
