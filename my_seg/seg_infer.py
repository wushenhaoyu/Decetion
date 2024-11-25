import os
import sys
from deploy.python.infer import Predictor
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)