from setuptools import setup


setup(name='ESIMtorch',
      version=1.0,
      url='https://github.com/coetaur0/ESIMtorch',
      license='Apache 2',
      author='Aurelien Coet',
      author_email='aurelien.coet19@gmail.com',
      description='Implementation in Pytorch of the ESIM model for NLI',
      packages=[
        'esimtorch',
        'esimtorch.model',
        'esimtorch.scripts'
      ],
      install_requires=[
        'numpy',
        'matplotlib',
        'torch'
      ])