# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution('alpha_pose').version


import alpha_pose.dataloader
from alpha_pose.SPPE.src.main_fast_inference import InferenNet
from alpha_pose.SPPE.src.main_fast_inference import InferenNet_fast
from alpha_pose.yolo.darknet import Darknet
from alpha_pose.yolo import util as yolo_utils

from alpha_pose.process_images import process_images
