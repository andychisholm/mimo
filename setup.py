from setuptools import setup, find_packages

__version__ = '0.1.0'
__pkg_name__ = 'mimo'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Mimo',
    author='Andrew Chisholm',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/andychisholm/mimo',
    entry_points = {},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires = [
        "numpy",
        "scipy",
        "ujson",
        "spacy",
        "tqdm",
        "pytorch"
    ],
    test_suite = __pkg_name__ + '.test'
)
