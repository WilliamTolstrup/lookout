from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lookout_tower'

setup(
    name=package_name,
    version='0.0.0',
    # Packages to export
    packages=[package_name],
    # Files we want to install, specifically launch files
    data_files=[
        # Install marker file in the package index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds/*'))),
        (os.path.join('share', package_name, 'description'), glob(os.path.join('description/*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config/*'))),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='william',
    maintainer_email='WilliamTolstrup@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
