# setup.py
from setuptools import setup, find_packages

setup(
    name='zorch',
    version='0.1.12',
    packages=find_packages(),
    description='Cupy-based tools for STARK proving',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vitalik Buterin',
    author_email='v@buterin.com',
    url='https://github.com/vbuterin/zorch',  # Replace with your GitHub repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
