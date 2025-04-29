# docs/pelicanconf.py

# Delete any existing output directory before generating

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = os.environ.get('READTHEDOCS_PROJECT_URL', 'http://localhost:8000')

# … the rest of your config …
PATH           = '.'
PAGE_PATHS     = ['.']
ARTICLE_PATHS  = []
PAGE_URL       = '{slug}.html'
PAGE_SAVE_AS   = '{slug}.html'
RELATIVE_URLS  = True
TIMEZONE       = 'Europe/Amsterdam'
DEFAULT_LANG   = 'en'
THEME          = 'notmyidea'
