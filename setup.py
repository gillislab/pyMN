import setuptools

with open('README.md'.'r') as fh:
	long_description = fh.read()

setuptools.setup(
	name='pyMetaNeighbor',
	version='0.1',
	author='Python Implementation of MetaNeighbor Algorithm for scRNAseq analysis',
	long_description=long_description,
	long_description_content_type='text/markdown',
	 )
