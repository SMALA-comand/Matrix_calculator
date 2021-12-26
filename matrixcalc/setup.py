import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrixcalc",
    packages=['matrixcalc'],
    version="0.0.1",
    description='user-friendly calculator + algorithms of various matrix operations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lev Pambukhchyan",
    author_email="leva200211@gmail.com",
    url="https://github.com/SMALA-comand/Matrix_calculator",
    install_requires=['numpy', 'sympy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
