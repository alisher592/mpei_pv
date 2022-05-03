import sys
import logging

logging.basicConfig(stream=sys.stderr)

sys.path.insert(0,'/var/www/mpei-pv')
#sys.path.append('/var/www/mpei-pv/venv/lib/python3.6/site-packages')


from main import app as application

application.secret_key = '4411'