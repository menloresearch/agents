from setuptools import setup, find_packages

setup(
    name="quarkflow",
    version="0.1.0",
    description="A lightweight library for creating and managing AI agent circuits",
    author="Menlo Deep Labs",
    author_email="info@menlodeep.com",
    packages=find_packages(),
    install_requires=[
        "bhumi>=0.1.0",
        "numpy>=1.20.0",
        "typing-extensions>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
