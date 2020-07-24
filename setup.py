from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


version = '0.0.1'


setup_requires = []

with open('requirements.txt') as f:
    install_requires = []
    for line in f:
        req = line.split('#')[0].strip()
        install_requires.append(req)


data_files = [
    'alpha_pose/yolo/cfg/tiny-yolo-voc.cfg',
    'alpha_pose/yolo/cfg/yolo-voc.cfg',
    'alpha_pose/yolo/cfg/yolo.cfg',
    'alpha_pose/yolo/cfg/yolov3-spp.cfg',
    'alpha_pose/yolo/cfg/yolov3.cfg',
]



setup(
    name='alpha_pose',
    version=version,
    description='A alpha pose library',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/alpha_pose',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'alpha_pose': data_files},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
