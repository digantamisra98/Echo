from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'echoAI',
    packages = setuptools.find_packages(),
    version = '0.1.1',  # Ideally should be same as your GitHub release tag varsion
    description = 'Python package containing all custom and SOTA mathematical backend algorithms used in Machine Learning.',
    author = 'Diganta Misra and Aleksandra Deis',
    author_email = 'mishradiganta91@gmail.com',
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/digantamisra98/Echo',
    download_url = 'https://github.com/digantamisra98/Echo/archive/0.1.1.tar.gz',
    keywords = ['machine learning', 'deep learning', 'algorithms', 'mathematics'],
    classifiers = [],
)
