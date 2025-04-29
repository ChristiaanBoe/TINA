# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''  # leave blank on RTD

# (we’re passing the pages folder on the CLI, so no PATH/PAGE_PATHS needed)

# disable “articles”
ARTICLE_PATHS = []

# Flatten every page to /<slug>.html
PAGE_URL     = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'

# So that internal links work on RTD
RELATIVE_URLS = True

TIMEZONE     = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'

THEME = 'notmyidea'
