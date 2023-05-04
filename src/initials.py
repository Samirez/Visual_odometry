import cv2  # type: ignore
import numpy as np
import glob
import OpenGL.GL as gl  # type: ignore
import collections
import g2o  # type: ignore
import ThreeDimViewer
import math
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from codetiming import Timer

np.set_printoptions(precision=4, suppress=True)

Feature = collections.namedtuple('Feature',
                                 ['keypoint', 'descriptor', 'feature_id'])
Match = collections.namedtuple('Match',
                               ['featureid1', 'featureid2',
                                'keypoint1', 'keypoint2',
                                'descriptor1', 'descriptor2',
                                'distance', 'color'])
Match3D = collections.namedtuple('Match3D',
                                 ['featureid1', 'featureid2',
                                  'keypoint1', 'keypoint2',
                                  'descriptor1', 'descriptor2',
                                  'distance', 'color',
                                  'point'])
MatchWithMap = collections.namedtuple('MatchWithMap',
                                      ['featureid1', 'featureid2',
                                       'imagecoord', 'mapcoord',
                                       'descriptor1', 'descriptor2',
                                       'distance'])
