"""Fallback setup.py for environments that don't support pyproject.toml."""

from setuptools import setup, find_packages

setup(
    name="sttai",
    version="0.1.0",
    description="Official Python SDK for STT.ai - Speech-to-Text API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="STT.ai",
    author_email="hello@stt.ai",
    url="https://stt.ai",
    project_urls={
        "Documentation": "https://stt.ai/docs/python-sdk",
        "Repository": "https://github.com/sttaigit/stt-python",
        "Issues": "https://github.com/sttaigit/stt-python/issues",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.20.0",
        "websocket-client>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="speech-to-text stt transcription whisper diarization api",
    license="MIT",
)
