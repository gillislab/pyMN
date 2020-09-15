import setuptools

from pathlib import Path
with open('README.md'.'r') as fh:
	long_description = fh.read()

setuptools.setup(
	name='pyMetaNeighbor',
	version='0.1.0',
	author='Ben Harris',
	author_email='bharris@cshl.edu',
	description='Python Implementation of MetaNeighbor Algorithm for scRNAseq analysis',
	long_description=long_description,
	long_description_content_type='text/markdown',
	install_requires=[l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()],
	packages=setuptools.find_packages(include=['pymn'])	
	 )
