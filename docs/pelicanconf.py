# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''        # keep blank on RTD

# ── IMPORTANT ────────────────────────────────────────────────────────────────
# We’re calling Pelican on docs/content/pages/, so tell it to treat **that root**
# as the page folder:
PAGE_PATHS    = ['.']
ARTICLE_PATHS = []   # disable articles entirely

# Flatten each page to /<slug>.html
PAGE_URL     = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'
# ────────────────────────────────────────────────────────────────────────────

RELATIVE_URLS = True
TIMEZONE      = 'Europe/Amsterdam'
DEFAULT_LANG  = 'en'
THEME         = 'notmyidea'
