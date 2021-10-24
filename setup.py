import setuptools

setuptools.setup(
    name="colorsort",
    version="0.1.0",
    url="https://github.com/rowland-208/colorsort",
    author="James Rowland",
    author_email="rowland.208@gmail.com",
    description="Perceptually sorted colors",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=['opencv-python>=4.5.3', 'colormath>=3.0', 'numpy>=1.19.5', 'ortools>=9.0', 'scipy>=1.5.4'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9 ',
        'Topic :: Scientific/Engineering'
    ],
)
