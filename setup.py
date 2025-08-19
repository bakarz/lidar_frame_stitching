from setuptools import setup

package_name = 'lidar_frame_stitching'


install_requires=[
    'numpy',
    'open3d',
    'scipy', 
    'pandas',
    'scikit-learn',
],

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haythem boubaker',
    maintainer_email='haythemboubaker18@email.com',
    description='LiDAR stitch node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stitch_node = lidar_frame_stitching.stitch_node:main',
        ],
    },
)

