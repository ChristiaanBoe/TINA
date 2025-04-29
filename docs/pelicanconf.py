# docs/pelicanconf.py
import os

# ========== Core Settings ==========
SITEURL = os.environ.get('READTHEDOCS_PROJECT_URL', 'http://localhost:8000')
PATH = 'content/pages'

OUTPUT_PATH = os.path.join(os.environ.get('READTHEDOCS_OUTPUT', ''), 'html')
# Required Read the Docs integration
DELETE_OUTPUT_DIRECTORY = True  # Let Pelican handle cleanup
AUTHOR = 'Your Name'
SITENAME = 'TINA Documentation'
TIMEZONE = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'




# Basic configuration
AUTHOR = 'Your Name'
SITENAME = 'TINA Documentation'
TIMEZONE = 'Europe/Amsterdam'
DEFAULT_LANG = 'en'
THEME = 'simple'  # Use built-in theme
# ========== Read the Docs Specific ==========
# Use environment variable for output path
READTHEDOCS_OUTPUT = os.environ.get('READTHEDOCS_OUTPUT', 'output')
OUTPUT_PATH = os.path.join(READTHEDOCS_OUTPUT, 'html')

# ========== Content Generation ==========
DELETE_OUTPUT_DIRECTORY = False  # Let Read the Docs handle cleanup
USE_FOLDER_AS_CATEGORY = True
DEFAULT_PAGINATION = 10

# ========== Theme & Styling ==========
THEME = 'simple'  # Built-in Pelican theme
STATIC_PATHS = ['static']
CSS_FILE = 'custom.css'  # Add custom CSS if needed

# ========== Feed Settings ==========
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# ========== Plugins & Markdown ==========
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

# ========== Navigation ==========
MENUITEMS = (
    ('Home', SITEURL),
    ('GitHub Repository', 'https://github.com/ChristiaanBoe/TINA'),
)

# ========== Read the Docs Integration ==========
# Set this to match your RTD project name
READTHEDOCS_PROJECT = 'tina-pages'

# ========== Optional Settings ==========
# Uncomment these if you need them
# PLUGIN_PATHS = ['plugins']
# PLUGINS = ['sitemap', 'neighbors']
# DISQUS_SITENAME = ''
# GOOGLE_ANALYTICS = ''
