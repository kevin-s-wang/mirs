import sys
sys.path.append('../mirs')

from mirs.conf.config import config

print(config.clip.get_model())