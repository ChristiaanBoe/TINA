# docs/pelicanconf.py

# Delete any existing output directory before generating
DELETE_OUTPUT_DIRECTORY = True

AUTHOR = 'Christiaan Boerkamp'
SITENAME = 'TINA Documentation'
SITEURL = ''

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
