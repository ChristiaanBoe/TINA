# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''  # keep blank on RTD

# where to find your content
PATH = 'content'

# only scan docs/content/pages/ for pages
PAGE_PATHS    = ['pages']
ARTICLE_PATHS = []        # no articles

# ensure each page is output at /<slug>.html
PAGE_URL      = '{slug}.html'
PAGE_SAVE_AS  = '{slug}.html'

# optional but recommended for RTD so internal links stay local
RELATIVE_URLS = True

TIMEZONE = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'

THEME = 'notmyidea'
