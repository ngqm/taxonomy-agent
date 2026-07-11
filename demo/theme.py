"""Visual design system — "The Journal" (Direction B).

Purely presentational. This module injects the app-wide look (fonts, Day/Night
palettes, and Streamlit-widget overrides) and provides HTML builders for the
bespoke catalogue components (specimen cards, the ledger stat strip,
distribution bars, the map legend). It contains no application logic and reads
no run data — callers pass already-loaded values in.

Editorial / scientific-catalogue aesthetic: Newsreader (serif display + figures)
+ Public Sans (body/labels), a warm "newsprint" (Day) / cool "slate-press"
(Night) theme, square corners, and a press-red / coral accent. Both themes share
one category-colour map so a category keeps its colour across every tab and chart.
"""
from __future__ import annotations

from html import escape as _esc

import streamlit as st

# ── Category colours (shared by cards, bars, and the corpus map) ─────────────
# One colour = one category, everywhere. Chosen at mid-luminance so they read on
# both the warm-paper and deep-ink canvases; the palette does NOT change between
# themes — only the surrounding neutrals do. 'other' is always neutral grey.
CATEGORY_COLORS: dict[str, str] = {
    "anthropomorphization": "#5a6ed6",  # indigo-blue (nudged off the teal)
    "sycophancy":           "#2fb0a0",  # teal-green (nudged off the blue)
    "brand_bias":           "#c79237",  # amber
    "user_retention":       "#a85fc0",  # violet
    "sneaking":             "#e15e39",  # tomato (nudged off the vermilion UI accent)
    "harmful_generation":   "#3f9f6b",  # green
    "polarizing_stance":    "#c74f8a",  # magenta
    "other":                "#a6a6ad",  # grey
}

# Deterministic fallback hues for categories not in the named map above (i.e.
# any corpus other than DarkBench). Same mid-luminance family so arbitrary runs
# still read as one system.
FALLBACK_PALETTE: list[str] = [
    "#5a6ed6", "#2fb0a0", "#c79237", "#a85fc0", "#e15e39",
    "#3f9f6b", "#c74f8a", "#6f7fd6", "#4a9d8e", "#b8863f",
    "#8f6fb5", "#c9705a", "#5aa06f", "#c05f92", "#7d94c4",
]

OTHER_GREY = CATEGORY_COLORS["other"]


# ── Theme tokens (light "Day" / dark "Night") ────────────────────────────────
# Two neutral+accent palettes: a warm "newsprint" Day and a cool "slate-press"
# Night. sRGB hex / rgba values with their oklch() equivalents in comments; the
# shared category-colour palette (above) is theme-independent and unchanged.
_THEMES: dict[str, dict[str, str]] = {
    "day": {  # "The Journal" — newsprint
        "canvas":       "#E4E1D8",  # oklch(0.910 0.013 92)
        "panel":        "#FBFAF6",
        "panel-2":      "#F2F0E9",  # frame / app bg  oklch(0.955 0.010 94)
        "sidebar-bg":   "#ECE9E1",  # oklch(0.934 0.011 90)
        "frame-border": "#C7C2B8",  # oklch(0.815 0.015 85)
        "card-border":  "#D3CEC4",  # oklch(0.853 0.015 85)
        "rule":         "#DAD6CD",  # oklch(0.877 0.013 87)
        "section-rule": "rgba(20,18,16,0.55)",
        "ink":          "#1A1712",  # oklch(0.206 0.011 81)
        "ink-strong":   "#0F0C08",  # oklch(0.157 0.010 76)
        "body":         "#403A33",  # oklch(0.352 0.014 72)
        "muted":        "#6E665D",  # oklch(0.515 0.017 71)
        "faint":        "#8B837A",  # oklch(0.614 0.016 71)
        "accent":       "#1E4C6E",  # ink blue
        "accent-soft":  "rgba(30,76,110,0.10)",
        "track":        "#E0DDD5",  # oklch(0.898 0.011 90)
        "code-bg":      "#EDEBE4",
    },
    "night": {  # "The Journal" — slate press
        "canvas":       "#0F1113",  # oklch(0.176 0.005 248)
        "panel":        "#1A1D20",  # card  oklch(0.229 0.007 248)
        "panel-2":      "#141619",  # frame oklch(0.199 0.007 258)
        "sidebar-bg":   "#0A0C0E",  # oklch(0.153 0.005 248)
        "frame-border": "#333A40",  # oklch(0.344 0.014 244)
        "card-border":  "#373E44",  # oklch(0.360 0.014 244)
        "rule":         "#2D3339",  # oklch(0.318 0.014 248)
        "section-rule": "rgba(111,168,204,0.42)",
        "ink":          "#E6E8EA",  # oklch(0.930 0.003 248)
        "ink-strong":   "#F1F3F5",  # oklch(0.963 0.003 248)
        "body":         "#AAB0B6",  # oklch(0.754 0.011 248)
        "muted":        "#8C939A",  # oklch(0.660 0.013 248) — raised for WCAG legibility on ink
        "faint":        "#767D84",  # oklch(0.586 0.014 248) — placeholders/quotes stay readable
        "accent":       "#6FA8CC",  # steel blue
        "accent-soft":  "rgba(111,168,204,0.16)",
        "track":        "#282C30",  # oklch(0.291 0.009 248)
        "code-bg":      "#16191C",
    },
}

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;"
    "1,6..72,400;1,6..72,500&"
    "family=Public+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');"
)

_SERIF = "'Newsreader', Georgia, 'Times New Roman', serif"
_SANS = "'Public Sans', system-ui, -apple-system, sans-serif"

# Theme-independent widget overrides. References var(--token) values set per
# theme by inject_theme(). Kept as a plain string so literal CSS braces need no
# f-string escaping.
_BASE_CSS = """
/* ── base canvas + typography ─────────────────────────────────────────── */
.stApp { background: var(--canvas) !important; color: var(--body); font-family: %(sans)s; }
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] { background: var(--canvas) !important; }
[data-testid="stHeader"] { background: transparent !important; }
/* Hide Streamlit's own top chrome (Deploy button, hamburger, rainbow bar) so
   the page reads as the catalogue, not a Streamlit app — but NOT the whole
   stToolbar: it also holds the collapsed-sidebar *expand* button, so hiding the
   toolbar left no way to reopen the sidebar once collapsed. Hide only the
   specific items and keep [data-testid="stExpandSidebarButton"] visible. */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stDecoration"] { display: none !important; }
/* The main column reads as a framed "printed page": lighter paper than the
   canvas margin, hairline border, soft shadow. Cards sit white on top of it. */
[data-testid="stMainBlockContainer"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--frame-border);
  box-shadow: 0 16px 46px -24px rgba(30, 22, 14, 0.42);
  padding: 30px 42px 46px !important;
  margin: 14px auto 30px !important;
  max-width: 1180px;
}
body, p, span, div, label, li, td, th, .stMarkdown,
[data-testid="stMarkdownContainer"] { font-family: %(sans)s; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stHeading"] {
  font-family: %(serif)s !important; color: var(--ink) !important;
  font-weight: 500; letter-spacing: 0;
}
.stApp h1 { font-size: 2.35rem; letter-spacing: -0.005em; }
/* Section heads (st.header / st.subheader) sit above a vermilion hairline. */
.stApp h2, .stApp h3 {
  border-bottom: 1px solid var(--section-rule);
  padding-bottom: 6px; margin-bottom: 12px; font-size: 1.6rem;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: var(--muted) !important; font-family: %(sans)s;
}

/* ── tabs: SERIF labels (16px), ink active, vermilion underline ────────── */
/* Streamlit 1.59 tabs are [data-testid="stTab"] inside [role="tablist"]. */
.stApp [role="tablist"] { border-bottom: 1px solid var(--card-border); gap: 26px; }
.stApp [data-testid="stTab"] p {
  font-family: %(serif)s !important; font-size: 17px !important; font-weight: 400;
  letter-spacing: 0; color: var(--muted) !important;
}
.stApp [data-testid="stTab"][aria-selected="true"] p { color: var(--ink) !important; }
.stApp [data-testid="stTab"][aria-selected="true"] { border-bottom-color: var(--accent) !important; }

/* ── buttons: square; primary = ink fill (cream text) ─────────────────── */
[data-testid^="stBaseButton"] { border-radius: 0 !important; font-family: %(sans)s !important; font-weight: 600; letter-spacing: 0.02em; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--ink) !important; color: var(--canvas) !important;
  border: 1px solid var(--ink) !important;
}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.14); }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--panel) !important; color: var(--ink) !important;
  border: 1px solid var(--card-border) !important;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── inputs / selects / textareas: square, themed bg, accent focus ─────── */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div,
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextInputRootElement"],
.stApp input, .stApp textarea {
  background-color: var(--panel) !important; border-radius: 0 !important;
  border-color: var(--card-border) !important; color: var(--ink) !important;
}
.stApp input, .stApp textarea, .stApp [data-baseweb="input"] input,
.stApp [data-baseweb="select"] div { font-family: %(serif)s !important; }
.stApp input::placeholder, .stApp textarea::placeholder { color: var(--faint) !important; }
.stApp [data-baseweb="select"] div { color: var(--ink) !important; }
.stApp [data-testid="stNumberInput"] button {
  background-color: var(--panel) !important; color: var(--ink) !important;
  border-color: var(--card-border) !important; border-radius: 0 !important;
}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
/* file-uploader dropzone reads as a dashed catalogue box on paper */
.stApp [data-testid="stFileUploaderDropzone"] { border: 1.5px dashed var(--card-border) !important; background: var(--panel) !important; border-radius: 0 !important; }
/* multiselect selected-value chips ("tags"): square them (matches the
   catalogue's square corners) and inset the value area, so the first chip is no
   longer flush against the border with white gaps showing through the rounded
   pill corners (read as the chip being "cut" on the left). */
/* Selectbox: the dropdown-arrow cell is painted with Streamlit's light base
   secondaryBackgroundColor — a shaded "caret cell" split off from the field in
   Day, and a bright cell in Night. Repaint the control wrapper on the panel
   colour so the select reads as one uniform field in both themes. */
.stApp [data-testid="stSelectbox"] div { background-color: var(--panel) !important; }
/* ...but keep the widget label itself on the canvas (the broad rule above would
   otherwise paint a panel-coloured box behind "TASK PRESET" etc.). */
.stApp [data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
.stApp [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * { background-color: transparent !important; }
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] { border-radius: 0 !important; }
/* The left inset must live on the label span, NOT the tag: the tag's own
   padding-left is swallowed by the value container's overflow clip, so the text
   rendered flush against the chip's left edge (which read as "cropped"). The
   label span carries the breathing room; the control padding insets the whole
   chip off the input border. */
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span[title] { padding-left: 8px !important; }
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div { padding-left: 6px !important; }
/* Sidebar widget labels + radio/checkbox option text = theme body colour
   (fixes labels/option text going invisible on the dark Night sidebar). */
.stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"], .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCheckbox"] label {
  color: var(--body) !important;
}
/* Main-column form labels read as UPPERCASE Public Sans kickers. */
.stApp [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
  text-transform: uppercase; letter-spacing: 0.14em; font-size: 10.5px;
  font-weight: 600; color: var(--muted) !important; font-family: %(sans)s !important;
}

/* ── sidebar: flat "spec-sheet" — no expander boxes, hairline kickers ──── */
[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; border-right: 1px solid var(--frame-border); }
/* Tighten the sidebar's vertical rhythm to the mockup's compact spec-sheet:
   less air between widgets, between spec rows, and between groups. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { gap: 0.55rem !important; padding-top: 11px; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding-top: 2px !important; }
/* Separate the section groups from each other (and from the "Configuration"
   heading) by more than the gap between fields inside a group, so the kickers
   read as group headers rather than just another tight row. */
[data-testid="stSidebar"] [data-testid="stExpander"] { margin-top: 16px !important; }
[data-testid="stSidebar"] [data-testid="stHeading"], [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
  border-bottom: none !important; padding-bottom: 0 !important; font-family: %(serif)s !important;
  color: var(--ink) !important; font-size: 1.2rem !important; margin-bottom: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border: none !important; background: transparent !important; box-shadow: none !important;
  border-radius: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.7rem !important;
  font-weight: 600; color: var(--muted) !important; font-family: %(sans)s !important;
  border-bottom: 1px solid var(--frame-border); padding: 0 0 6px 0 !important;
  min-height: 0 !important; align-items: flex-end !important; line-height: 1.2 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: 0.7rem !important; letter-spacing: 0.16em; line-height: 1.2 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
[data-testid="stSidebar"] [data-testid="stExpander"] summary [data-testid="stIconMaterial"] { display: none !important; }
/* The collapse-chevron sits in its own 20px flex slot ahead of the label; the
   icon glyph above is hidden but the slot + its 8px gap still indent every
   kicker ~28px past the "Configuration" heading. Drop the slot and the gap so
   kickers left-align with the heading. (Label text is a div, not a span, so
   hiding the summary's inner span hits only the icon slot.) */
[data-testid="stSidebar"] [data-testid="stExpander"] summary > span { gap: 0 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary > span > span { display: none !important; }
/* hide the +/- steppers on sidebar number inputs; right-align the serif value */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button { display: none !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] input { text-align: right; }
/* fully flatten the sidebar groups + make numeric fields borderless spec rows */
[data-testid="stSidebar"] [data-testid="stExpander"] summary { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="base-input"] {
  border: none !important; background: transparent !important; box-shadow: none !important;
}
.side-numlabel { font-family: %(sans)s; font-size: 12.5px; color: var(--body); display: block; padding-top: 7px; }
/* keep every sidebar field label (selectbox, text input, slider) at the same
   reasonable size as the numeric spec-row labels, so no label is visibly
   larger than its neighbours within a section. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 12.5px !important; }
/* Hide the "?" help-tooltip icons next to field labels in the MAIN content
   (the Run form explains itself with placeholders + captions), but keep them
   in the sidebar so every config field carries an explanation. */
.stApp [data-testid="stMain"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] { display: inline-flex !important; }
/* "?" affordance on the numeric spec-rows, whose collapsed labels can't carry
   Streamlit's own help icon. `title` supplies the hover text. */
.side-help {
  display: inline-block; margin-left: 5px; width: 13px; height: 13px;
  line-height: 12px; text-align: center; font-size: 9px; font-weight: 700;
  color: var(--muted); border: 1px solid var(--frame-border);
  border-radius: 50%%; cursor: help; vertical-align: middle; user-select: none;
}

/* ── segmented controls (Day|Night in sidebar; Source on Run) ──────────── */
.stApp [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
  border-radius: 0 !important; border: 1px solid var(--card-border) !important;
  background: var(--panel) !important; font-family: %(sans)s !important;
}
.stApp [data-testid="stButtonGroup"] button[data-variant="segmented_control"] p { color: var(--body) !important; }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--ink) !important; border-color: var(--ink) !important;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--canvas) !important; }
.st-key-source_seg [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--ink) !important; border-color: var(--ink) !important;
}
.st-key-source_seg [data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--canvas) !important; }

/* ── the "Run the demo" call-out (keyed container in run.py) ────────────── */
/* Warm-paper panel with a vermilion left-rule (matching the NEW? callout) and
   a single filled-vermilion primary button — vermilion is the one "act" colour
   in the system, so the recommended action wears it (not an off-palette green). */
.st-key-demo_cta { border: 1px solid var(--card-border); border-left: 3px solid var(--accent); background: var(--panel-2); padding: 15px 18px 17px; margin-bottom: 12px; }
.st-key-demo_cta [data-testid^="stBaseButton"] { background: var(--accent) !important; border: 1px solid var(--accent) !important; color: #ffffff !important; }
.st-key-demo_cta [data-testid^="stBaseButton"]:hover { filter: brightness(1.08); }

/* ── Corpus-map "plate": a bordered white frame around the map + legend ── */
.st-key-map_plate { border: 1px solid var(--card-border); background: var(--panel); padding: 14px; margin-bottom: 4px; }

/* ── metrics (History / Run / trace): serif figures, small-caps labels ── */
[data-testid="stMetricValue"] { font-family: %(serif)s !important; color: var(--ink) !important; font-weight: 500; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  font-family: %(sans)s !important; text-transform: uppercase; letter-spacing: 0.12em;
  font-size: 0.68rem !important; color: var(--muted) !important;
}

/* ── expanders / alerts / code / dataframe ────────────────────────────── */
[data-testid="stExpander"] { border: 1px solid var(--card-border) !important; border-radius: 0 !important; background: var(--panel) !important; }
[data-testid="stExpander"] summary { color: var(--ink); }
/* Pin the main-area expander header to the theme in EVERY interactive state.
   Streamlit derives its hover/active/focus tint from the light config.toml base
   theme, so in Night that pale tint can wash the header background and hide the
   light header text (clicking a run row or "Final classification prompt" makes
   the text vanish). Force dark panel + light ink here; scoped to the main area
   so the flattened sidebar expanders are untouched. */
[data-testid="stMain"] [data-testid="stExpander"] summary { background: var(--panel) !important; color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stExpander"] summary:hover,
[data-testid="stMain"] [data-testid="stExpander"] summary:focus,
[data-testid="stMain"] [data-testid="stExpander"] summary:focus-visible,
[data-testid="stMain"] [data-testid="stExpander"] summary:active,
[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary { background: var(--track) !important; color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stExpander"] summary p,
[data-testid="stMain"] [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] * { color: var(--ink) !important; }
/* Clamp long expander headers (e.g. History run rows carrying the goal text) to
   a tidy two-line block so cards don't vary in height with truncated overflow. */
[data-testid="stMain"] [data-testid="stExpander"] summary p {
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
/* Flatten the secondary "type a run directory path" expander into a light,
   link-like toggle so it doesn't read as a second bordered field beside the
   run selectbox (Inspect › Run under inspection). */
.st-key-run_path_adv [data-testid="stExpander"] { border: none !important; background: transparent !important; }
.st-key-run_path_adv [data-testid="stMain"] [data-testid="stExpander"] summary,
.st-key-run_path_adv [data-testid="stExpander"] summary { background: transparent !important; padding-left: 0 !important; }
.st-key-run_path_adv [data-testid="stExpander"] summary p { color: var(--muted) !important; font-size: 0.82rem; }
.stApp [data-testid="stAlert"], .stApp [data-testid="stAlert"] > div,
.stApp [data-testid="stAlertContainer"] {
  background-color: var(--panel-2) !important; border-radius: 0 !important;
}
.stApp [data-testid="stAlert"] {
  border: 1px solid var(--card-border) !important;
  border-left: 3px solid var(--accent) !important;
}
.stApp [data-testid="stAlert"] * { color: var(--body) !important; }
.stApp [data-testid="stCode"], .stApp pre { background: var(--code-bg) !important; border: 1px solid var(--card-border) !important; border-radius: 0 !important; }
.stApp code { background: var(--code-bg) !important; color: var(--body) !important; border-radius: 0 !important; padding: 0.05em 0.34em; font-size: 0.86em; }
.stApp pre code { background: transparent !important; padding: 0 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--card-border) !important; }
hr { border-color: var(--rule); }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }

/* ── bespoke component utility classes ────────────────────────────────── */
.page-eyebrow { font-family: %(sans)s; font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 2px 0 -2px; }
/* Masthead title — Newsreader display (weight 500). Rendered as a bespoke
   element (not st.title) because st.title wraps its text in a span/div that the
   generic `span,div {sans}` rule would otherwise force back to Public Sans. */
.page-title { font-family: %(serif)s; font-weight: 500; font-size: 38px; line-height: 1.05; letter-spacing: -0.005em; color: var(--ink); margin: 6px 0 6px; }
.page-subtitle { font-family: %(serif)s; font-size: 17px; line-height: 1.5; color: var(--body); max-width: 780px; margin: 4px 0 8px; }
.kicker { font-family: %(sans)s; font-size: 10.5px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 6px 0 8px; }
.fig-cap { font-family: %(serif)s; font-size: 13.5px; color: var(--body); margin: 10px 0 30px; }
.fig-cap .runin { font-family: %(sans)s; font-style: normal; font-weight: 600; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink); }
.other-note { font-family: %(sans)s; font-size: 12px; color: var(--muted); margin: 2px 0 8px; display: flex; align-items: center; gap: 8px; }
.other-note .sw { width: 10px; height: 10px; background: %(grey)s; display: inline-block; flex: none; }
.step-head { display: flex; align-items: baseline; gap: 12px; margin: 10px 0 10px; }
.step-num { font-family: %(serif)s; font-size: 19px; color: var(--accent); width: 20px; flex: none; }
.step-label { font-family: %(serif)s; font-weight: 500; font-size: 19px; color: var(--ink); }
.new-banner { display: flex; align-items: baseline; gap: 12px; background: var(--accent-soft); border: 1px solid var(--section-rule); border-left: 3px solid var(--accent); padding: 12px 16px; margin: 4px 0 16px; }
.new-banner .tag { font-family: %(sans)s; font-size: 10.5px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); flex: none; }
.new-banner .msg { font-family: %(sans)s; font-size: 13px; line-height: 1.5; color: var(--body); }
.demo-serif { font-family: %(serif)s; font-size: 14px; line-height: 1.45; color: var(--body); margin-bottom: 8px; }

/* --- Narrow screens / mobile: stack column layouts (stat strips, card
   grids) that otherwise clip off the right edge, and tighten the frame.
   Scoped to the main area so the sidebar drawer is untouched. Literal
   percents are doubled because this string is %%-formatted below. */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 50px 16px 32px !important; margin: 6px auto 18px !important; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.6rem !important; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100%% !important; width: 100%% !important; min-width: 100%% !important;
  }
  .page-title { font-size: 30px !important; line-height: 1.08 !important; }
  .page-subtitle { font-size: 15.5px !important; }
}
""" % {"sans": _SANS, "serif": _SERIF, "grey": OTHER_GREY}


def inject_theme(theme: str = "day") -> None:
    """Inject the fonts, the selected theme's CSS variables, and all widget
    overrides. Safe to call once per rerun; category colours are theme-
    independent so only the neutral tokens change between Day and Night."""
    tokens = _THEMES.get(theme, _THEMES["day"])
    vars_block = "\n".join(f"  --{k}: {v};" for k, v in tokens.items())
    css = (
        "<style>\n"
        + _FONT_IMPORT + "\n"
        + ":root {\n" + vars_block + "\n}\n"
        + _BASE_CSS
        + "\n</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


# ── HTML component builders (catalogue "specimens") ──────────────────────────

def _titlecase(slug: str) -> str:
    """'brand_bias' -> 'Brand bias' (sentence case; underscores to spaces) for a
    human-readable serif headline. The exact machine slug is shown separately so
    no information is lost."""
    s = str(slug).replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else str(slug)


def gallery_card_html(num: int, name: str, count, desc: str, quote: str,
                      color: str) -> str:
    """One taxonomy "specimen" card: catalogue number, square colour chip,
    serif name + count, definition, and a blockquote representative item with a
    left rule in the category's colour. Square corners; neutrals from CSS vars."""
    _slug = _esc(str(name))
    _disp = _esc(_titlecase(name))
    parts = [
        # No fixed height / bottom margin: the card is a CSS-grid item (see
        # gallery_grid_html) and stretches to its row's height so every card in
        # a row is equal-sized and aligned; the grid gap handles spacing.
        '<div style="border:1px solid var(--card-border);background:var(--panel);'
        'padding:16px 18px;display:flex;flex-direction:column;">',
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:10px;">',
        # Catalogue number + the exact machine slug as a small mono kicker; the
        # serif headline below carries the human-readable (title-cased) name so
        # the display face never has to render an underscore.
        f'<span style="font-family:{_SANS};font-size:10px;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:var(--faint);font-weight:600;overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap;min-width:0;">Nº {num:02d}'
        f'<span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        f'letter-spacing:0;text-transform:none;color:var(--muted);"> · {_slug}</span></span>',
        f'<span style="width:11px;height:11px;background:{color};flex:none;display:inline-block;"></span>',
        '</div>',
        '<div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;">',
        f'<span style="font-family:{_SERIF};font-weight:500;font-size:22px;'
        f'line-height:1.1;color:var(--ink);">{_disp}</span>',
        f'<span style="font-family:{_SERIF};font-size:24px;color:var(--ink);'
        f'flex:none;line-height:1;">{count}</span>',
        '</div>',
        f'<div style="font-family:{_SANS};font-size:9.5px;letter-spacing:0.16em;'
        f'text-transform:uppercase;color:var(--faint);text-align:right;margin-top:1px;">items</div>',
    ]
    if desc:
        parts.append(
            f'<div style="font-family:{_SANS};font-size:12.5px;line-height:1.5;'
            f'color:var(--body);margin-top:11px;">{_esc(desc)}</div>'
        )
    if quote:
        parts.append(
            # margin-top:auto anchors the specimen to the card's bottom so
            # short cards don't strand blank space (cards are equal-height).
            '<div style="border-top:1px solid var(--rule);margin-top:auto;padding-top:11px;">'
            f'<div style="font-family:{_SANS};font-size:9px;letter-spacing:0.18em;'
            f'text-transform:uppercase;color:var(--faint);margin-bottom:7px;font-weight:600;">'
            'Representative specimen</div>'
            f'<blockquote style="margin:0;border-left:2px solid {color};padding-left:12px;'
            f'font-family:{_SERIF};font-style:italic;font-size:14px;line-height:1.5;'
            f'color:var(--muted);">{_esc(quote)}</blockquote>'
            '</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def gallery_grid_html(cards: list[str], ncol: int = 3) -> str:
    """Lay out specimen cards (from gallery_card_html) in an N-column CSS grid.
    A real grid — not st.columns — so every card in a row stretches to the same
    height (align-items:stretch) and rows line up, matching the mockup's
    `grid-template-columns:repeat(N,1fr)` catalogue gallery."""
    # auto-fit + minmax keeps the ~ncol-column catalogue on wide screens but
    # wraps to fewer (down to one) columns on narrow/mobile viewports instead
    # of clipping cards off the right edge. min(100%, ..) avoids overflow when
    # the container is narrower than the track's ideal width.
    _min = max(200, 900 // max(1, ncol))  # ~280px for ncol=3
    return (
        f'<div style="display:grid;'
        f'grid-template-columns:repeat(auto-fit,minmax(min(100%, {_min}px),1fr));'
        f'gap:16px;margin-bottom:14px;">' + "".join(cards) + '</div>'
    )


def stat_ledger_html(cells: list[dict], value_size: int = 50) -> str:
    """A single ruled "ledger" strip of N cells. Each cell dict:
      {label, value(HTML-safe str/number), sub?(str), chips?(list[hex]), title?(str)}
    `value` is inserted verbatim (callers format it); label/sub/title are escaped.
    `chips` renders a swatch strip in place of `sub` (used for the category tally).
    `value_size` is the figure px (50 for the headline ledger, 32 for the Cost row)."""
    out = [
        f'<div style="display:grid;'
        f'grid-template-columns:repeat(auto-fit,minmax(min(100%, 150px),1fr));'
        'gap:0;border:1px solid var(--card-border);background:var(--panel);margin:6px 0 10px;">'
    ]
    last = len(cells) - 1
    for i, c in enumerate(cells):
        border_r = "" if i == last else "border-right:1px solid var(--rule);"
        title_attr = f' title="{_esc(str(c["title"]))}"' if c.get("title") else ""
        cell = [f'<div style="padding:15px 18px;{border_r}"{title_attr}>']
        cell.append(
            f'<div style="font-family:{_SANS};font-size:10px;letter-spacing:0.16em;'
            f'text-transform:uppercase;color:var(--muted);font-weight:600;">{_esc(str(c["label"]))}</div>'
        )
        cell.append(
            f'<div style="font-family:{_SERIF};font-weight:500;font-size:{value_size}px;'
            f'line-height:1;color:var(--ink-strong);margin-top:6px;">{c["value"]}</div>'
        )
        if c.get("chips"):
            chips = "".join(
                f'<span style="width:14px;height:5px;background:{col};'
                'display:inline-block;margin-right:3px;"></span>'
                for col in c["chips"]
            )
            cell.append(
                '<div style="border-top:1px solid var(--rule);margin-top:10px;padding-top:8px;">'
                f'{chips}</div>'
            )
        elif c.get("sub"):
            cell.append(
                '<div style="border-top:1px solid var(--rule);margin-top:10px;padding-top:6px;'
                f'font-family:{_SANS};font-size:11px;color:var(--faint);">{_esc(str(c["sub"]))}</div>'
            )
        cell.append('</div>')
        out.append("".join(cell))
    out.append('</div>')
    return "".join(out)


def section_header_html(title: str, meta: str = "") -> str:
    """Editorial section header: serif title LEFT + muted meta RIGHT on one
    baseline row, above a vermilion hairline. Matches the mockup's section
    treatment (used instead of st.subheader + a separate caption)."""
    return (
        '<div style="display:flex;align-items:baseline;justify-content:space-between;'
        'border-bottom:1px solid var(--section-rule);padding-bottom:8px;margin:6px 0 18px;">'
        f'<span style="font-family:{_SERIF};font-weight:500;font-size:26px;'
        f'color:var(--ink);line-height:1.1;">{_esc(str(title))}</span>'
        f'<span style="font-family:{_SANS};font-size:11.5px;color:var(--muted);'
        f'white-space:nowrap;padding-left:16px;">{_esc(str(meta))}</span>'
        '</div>'
    )


def distribution_bars_html(pairs, cmap: dict, compact: bool = False) -> str:
    """Horizontal catalogue bars: Public Sans label gutter, square colour-filled
    track, serif count. `pairs` is an ordered iterable of (name, count). Pass
    `compact=True` for the tighter Compare-card variant (narrower gutter/bars)."""
    pairs = list(pairs)
    maxc = max((v for _, v in pairs), default=1) or 1
    if compact:
        cols, gap, mb, lab_sz, trk, num_sz = "120px 1fr 34px", "10px", "7px", "10.5px", "13px", "14px"
    else:
        cols, gap, mb, lab_sz, trk, num_sz = "150px 1fr 46px", "14px", "9px", "12px", "15px", "16px"
    rows = ['<div style="margin:4px 0 10px;">']
    for name, v in pairs:
        col = cmap.get(name, OTHER_GREY)
        w = max(1.5, v / maxc * 100)
        rows.append(
            f'<div style="display:grid;grid-template-columns:{cols};'
            f'align-items:center;gap:{gap};margin-bottom:{mb};">'
            f'<span style="font-family:{_SANS};font-size:{lab_sz};color:var(--body);'
            'text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
            f'{_esc(str(name))}</span>'
            f'<div style="height:{trk};background:var(--track);overflow:hidden;">'
            f'<div style="height:100%;width:{w:.1f}%;background:{col};"></div></div>'
            f'<span style="font-family:{_SERIF};font-size:{num_sz};color:var(--body);">{v}</span>'
            '</div>'
        )
    rows.append('</div>')
    return "".join(rows)


def legend_html(order, cmap: dict) -> str:
    """Map legend column: square swatch + serif category name, hairline rows."""
    order = list(order)
    last = len(order) - 1
    rows = [
        f'<div style="font-family:{_SANS};font-size:10px;letter-spacing:0.16em;'
        'text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-weight:600;">Legend</div>'
    ]
    for i, name in enumerate(order):
        col = cmap.get(name, OTHER_GREY)
        border = "" if i == last else "border-bottom:1px solid var(--rule);"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:9px;padding:5px 0;{border}">'
            f'<span style="width:10px;height:10px;background:{col};flex:none;display:inline-block;"></span>'
            f'<span style="font-family:{_SERIF};font-size:13px;color:var(--body);">{_esc(str(name))}</span>'
            '</div>'
        )
    return "".join(rows)


def chip_color_css(cmap: dict) -> str:
    """A per-run <style> block that tints each selected multiselect *filter* chip
    with its own category colour, so the chips echo the swatches on the cards and
    bars (one colour = one category, everywhere). Each rule matches the BaseWeb
    tag whose aria-label begins with the category name — chips that aren't
    category names (e.g. a directory path in the History multiselect) never match
    and keep the neutral accent. Purely presentational; reads no run data."""
    rules = []
    for name, col in (cmap or {}).items():
        if not name:
            continue
        esc = str(name).replace("\\", "\\\\").replace('"', '\\"')
        rules.append(
            f'.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"]'
            f'[aria-label^="{esc}, "]{{background:{col} !important;'
            f'border-color:{col} !important;}}'
        )
    return "<style>" + "".join(rules) + "</style>" if rules else ""


__all__ = [
    "CATEGORY_COLORS", "FALLBACK_PALETTE", "OTHER_GREY",
    "inject_theme", "gallery_card_html", "gallery_grid_html", "stat_ledger_html",
    "section_header_html", "distribution_bars_html", "legend_html", "chip_color_css",
]
