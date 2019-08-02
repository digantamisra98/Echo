from setuptools import setup

setup(name='Echo',
      version='1.0',
      author='Diganta Misra, Aleksandra Deis',
      author_email='mishradiganta91@gmail.com',
      url='https://github.com/digantamisra98/Echo',
      description='Python package containing SOTA mathematical backend algorithms used in Machine Learning. The initial release.',
      packages=['Echo', 'Echo.Activation', 'Echo.Activation.Torch', 'Echo.Activation.Keras'],
      zip_safe=False)
