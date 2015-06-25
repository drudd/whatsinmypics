import sys
sys.path.insert(0, '/whatsinmypics/')
sys.path.insert(1, '/whatsinmypics/caffe/python')
import os
print os.environ['LD_LIBRARY_PATH']
from whatsinmypics import app as application
