# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''        # keep blank on RTD

# ── Crucial fix ───────────────────────────────────────────────────────────────
# You’re passing the 'docs/content/pages' folder to Pelican on the CLI,
# so set your content path to the current directory:
PATH           = '.'
# ──────────────────────────────────────────────────────────────────────────────

# Only treat files in the content root as pages
PAGE_PATHS    = ['.']
ARTICLE_PATHS = []        # no articles

# Flatten every page to /<slug>.html
PAGE_URL     = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'

# Use relative URLs so links work on RTD
RELATIVE_URLS = True

TIMEZONE     = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'

THEME = 'notmyidea'
