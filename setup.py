from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("CFGpy/_version.py", "r") as f:
    exec(f.read())  # load version number from the package without using risky imports
setup(
    name='CFGpy',
    version=__version__,
    author='ComDePri Lab',
    author_email='roy.gutglick@mail.huji.ac.il',
    description='A python package for handling Creative Foraging Game behavioral data and numerical simulations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ComDePri/CFGpy',
    project_urls={},
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    package_data={'': ["behavioral/default_config.json"]},
    include_package_data=True
)
