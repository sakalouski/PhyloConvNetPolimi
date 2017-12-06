from setuptools import setup, find_packages

setup(name='PconvNetPolimi',
      version="0.0.1",
      description='Generation and investigation of gene distance matrices',
      author='Luca Nanni, Artur Sakalouski',
      packages=find_packages(exclude=("tests", "data", "scripts")),
      install_requires=['numpy', 'pandas', 'sklearn', 'imblearn', 'keras', 'tensorflow'],
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
             'Topic :: Scientific/Engineering :: Information Analysis',
            'Programming Language :: Python :: 3'
      ],
      keywords="genomics big data pandas python",
      python_requires='>=3',
      include_package_data=True,
      zip_safe=False)
