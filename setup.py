import os
from setuptools import setup

def read(shortname):
    filename = os.path.join(os.path.dirname(__file__), shortname)
    with open(filename, encoding='utf-8') as f:
        contents = f.read()
    return contents

setup(
    name="trish",
    version="0.0.1",
    author="multun",
    scripts=['trish', 'trish_batch'],
    description="A cheating detection tool.",
    keywords="cheating",
    url="https://github.com/multun/trish",
    packages=['libtrish'],
    python_requires='>= 3.6',
    install_requires=[],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
    ],
)
