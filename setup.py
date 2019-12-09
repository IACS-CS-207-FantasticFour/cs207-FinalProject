import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deltapi-sjmoonny",
    version="0.0.6",
    author="Anna Davydova, Dan Cox, Valentina Toll Villagra, Stephen Moon",
    author_email="stephenmoon@college.harvard.edu",
    description="DeltaPI Package for CS207",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IACS-CS-207-FantasticFour/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)