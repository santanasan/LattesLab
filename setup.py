from setuptools import setup

setup(name='LattesLab',
      version='0.10.3',
      description='Set of tools to generate data products originated by Lattes CV files.',
      url='https://github.com/tcodingprojects/LattesLab',
      author='Thiago Santana',
      author_email='thiagoluis@gmail.com',
      license='MIT',
      packages=['LattesLab'],
	  install_requires=[
          'wordcloud',
		  'scipy',
		  'datetime',
		  'matplotlib',
		  'sklearn',
		  'networkx',
		  'numpy',
		  'pandas',
		  'scikit-fuzzy'
      ],
	  setup_requires=[
		  'matplotlib',
		  'networkx',
		  'scipy',
		  'numpy',
		  'pandas'
	  ],
      zip_safe=False)