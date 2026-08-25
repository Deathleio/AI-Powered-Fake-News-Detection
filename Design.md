# Design.md — Visual Identity

Source of truth: `frontend/style.css` (CSS custom properties / design tokens).
Keep components referencing these tokens; don't re-invent colors inline.

---

## 1. Theme Vibe
**"Warm Editorial, Clear & Trustworthy."** Sand/off-white backdrop, terracotta primary for warmth, gentle green/red verdict accents (soft, not alarm-red). Rounded, calm, mobile-first.

---

## 2. Color Palette (`:root` tokens)

| Token | Value | Usage |
| --- | --- | --- |
| `--primary` | `#9a3412` | Terracotta / burnt ochre — brand, icons, headings accent |
| `--primary-hover` | `#7c2d12` | Primary hover |
| `--primary-light` | `#ffedd5` | Rationale box bg |
| `--primary-border` | `#fed7aa` | Rationale box border |
| `--bg-page` | `#fbf9f5` | Warm off-white / sand page bg |
| `--bg-card` | `#ffffff` | Card bg |
| `--bg-subtle` | `#f5f1ea` | Parchment tint (badges) |
| `--text-main` | `#292524` | Warm charcoal text |
| `--text-muted` | `#78716c` | Warm stone secondary |
| `--text-light` | `#a8a29e` | Tertiary / hints |
| `--border-color` | `#e7e5e4` | Soft stone borders |
| `--border-subtle` | `#f5f5f4` | Subtle dividers |

### Verdict accents (real = truth, fake = caution — soft, no alarm)

| Token | Value |
| --- | --- |
| `--real-text` `#1b5e20` / `--real-bg` `#e8f5e9` / `--real-border` `#c8e6c9` / `--real-accent` `#388e3c` | Sage/green |
| `--fake-text` `#991b1b` / `--fake-bg` `#fef2f2` / `--fake-border` `#fecaca` / `--fake-accent` `#dc2626` | Soft crimson |

---

## 3. Shadows & Radii

- Radius: `--radius-sm 8px`, `--radius-md 12px`, `--radius-lg 18px`, `--radius-full 9999px` (pills/badges).
- Shadows: subtle warm-toned elevation (`--shadow-sm/md/lg` driven by `rgba(41,37,36, …)`).

---

## 4. Typography

- **Family:** `'Plus Jakarta Sans'`, fallback `-apple-system, 'Segoe UI', Roboto, …` (loaded via Google Fonts in `index.html`).
- **Weights:** 400, 500, 600, 700, 800. Use 600/700 for headings and labels, 800 for logo/CTAs.
- **Base:** `html{font-size:16px}`, `body{line-height:1.6}`.
- **Scale (`rem`):**
  - `.logo` 1.25rem (1.1 rem mobile) · `.card h2` ~1.4rem · `.section-block h4` 0.9rem
  - body copy ~0.95rem · `.small-text` 0.78rem → 0.8rem footer
  - `.status-indicator` 0.78rem · `.chip` 0.78rem
- **Heading accents:** section headers use an icon in `--primary` + bold text; letter-spacing tightened on logo (`-0.3px`).

---

## 5. Layout & Components

- **Container:** max-width `1160px`, centered, `padding: 0 20px` (14px mobile).
- **Navbar:** sticky, white, bottom border, soft shadow; right-aligned live status pill.
- **Two-panel grid:** Input card + Results card (stacked to one column ≤860px).
- **Cards:** `--bg-card` white, radius-md, soft shadow, generous padding.
- **Forms:** labeled title input + body textarea, primary submit + secondary "Clear".
- **Result block:** verdict banner (bg = real/fake tint), confidence badge, animated progress meter, saliency chip grid, AI rationale box, annotated text panel.

### Signals UI
- **Status pill** green dot "online" / amber/grey "connecting".
- **Preset buttons:** bordered pill buttons; real preset = green accent, fake preset = crimson accent.
- **Chips:** `--fake` (red tint border) vs `--real` (green tint border) with `+weight`.

---

## 6. Motion & Micro-interactions
- `scroll-behavior: smooth`.
- `.progress-fill` animates width `0.5s cubic-bezier(0.4,0,0.2,1)`.
- Buttons hover → primary shade; tap highlight removed (`-webkit-tap-highlight-color: transparent`).
- Avoid aggressive animations; keeps framing calm/trustworthy.

---

## 7. Accessibility & Responsive
- High contrast between text (`#292524`) and sand bg.
- Breakpoint `≤860px`: single column, tighter padding, stacked verdict banner.
- `viewport maximum-scale=5.0` — enables pinch-zoom accessibility.
- All interactive elements reachable by keyboard / touch.

> ⚠️ If any new UI omits these tokens or invents new palette, default back to the tokens above. See `rules.md` section 5 (boundaries).