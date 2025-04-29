# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''

# where to find your content
PATH = 'content'

# include both content/ and content/pages/ as top‐level pages
PAGE_PATHS    = ['.', 'pages']

# disable Pelican “articles” entirely
ARTICLE_PATHS = []

# ────────────────────────────────────────────────────────────────────────────
# ensure every page ends up at /<slug>.html instead of /pages/<slug>.html
PAGE_URL      = '{slug}.html'
PAGE_SAVE_AS  = '{slug}.html'
# ────────────────────────────────────────────────────────────────────────────

TIMEZONE = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'

THEME = 'notmyidea'
