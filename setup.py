import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='propagsim',
    version='0.2',
    author='Manuel Capel',
    author_email='manuel.capel@parcoor.com',
    description='Implementation of CAST propagation model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/parcoor/py-propagsim',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)