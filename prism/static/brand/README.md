# Brand assets

Drop the official Samsung PRISM logo here as **`logo.svg`** (preferred) or
`logo.png`. Nothing else is needed — `prism/__init__.py:find_brand_logo()`
detects it and every page picks it up automatically: nav, footer and favicon.

Accepted filenames, in order of preference:

    logo.svg    logo.png    logo.webp    logo.jpg

Until a file is present, the templates render a typographic "SAMSUNG PRISM"
lockup in Samsung blue instead. That is a deliberate fallback, not a
placeholder graphic: an invented mark that looks official is worse than honest
type. Nothing here imitates a Samsung trademark.

## Notes

- The previous site hot-linked the Samsung wordmark from
  `upload.wikimedia.org`. That is fragile and against Wikimedia's hotlinking
  policy — self-host the asset here instead.
- On dark surfaces the logo is rendered with `filter: brightness(0) invert(1)`
  (class `.brand-logo`), so a dark-on-transparent SVG will show correctly.
  A logo with baked-in white already will need that rule adjusted.
- Samsung blue `#1428A0` is reserved for brand identity. The product UI uses
  the Auralis palette (indigo `#4F46E5`, cyan `#06B6D4`) so brand colour is
  never confused with a data colour or a risk band.
