from distutils.core import setup

setup(name='nn_homology',
      version='0.1',
      description='Computation graph homology package for Pytorch',
      author='Thomas Gebhart',
      author_email='gebhart@umn.edu',
      license='MIT',
      url='https://github.com/tgebhart/nn_homology',
      packages=['nn_homology',],
      long_description=open('README.md').read(),
     )
