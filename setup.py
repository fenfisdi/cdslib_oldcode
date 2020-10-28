import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cdslib',
    version='0.1.0',
    description='Contagious Disease Simulation - Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Camilo-HG/cdslib',
    author='Camilo Hincapié Gutiérrez',
    author_email='camilo.hincapie@udea.edu.co',
    license='gplv3',
    packages=setuptools.find_packages(),
    zip_safe=False
    )