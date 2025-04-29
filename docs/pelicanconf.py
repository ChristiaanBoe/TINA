# docs/pelicanconf.py

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''

# 1) Pelican’s root is docs/, so you look in docs/content/
PATH = 'content'

# 2) Only scan docs/content/pages/ for pages
PAGE_PATHS    = ['pages']
ARTICLE_PATHS = []        # no articles

# 3) Flatten URLs so each page is /<slug>.html
PAGE_URL      = '{slug}.html'
PAGE_SAVE_AS  = '{slug}.html'

# 4) Use relative URLs on RTD so internal links work
RELATIVE_URLS = True

TIMEZONE = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'

THEME = 'notmyidea'
