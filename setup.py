# -*- coding: utf-8 -*-
"""Installer for the collective.vectorsearch package."""

from setuptools import find_packages, setup

long_description = "\n\n".join(
    [
        open("README.rst").read(),
        open("CONTRIBUTORS.rst").read(),
        open("CHANGES.rst").read(),
    ]
)


setup(
    name="collective.vectorsearch",
    version="1.0a1",
    description="LLM vector search on Plone",
    long_description=long_description,
    # Get more from https://pypi.org/classifiers/
    classifiers=[
        "Environment :: Web Environment",
        "Framework :: Plone",
        "Framework :: Plone :: Addon",
        "Framework :: Plone :: 6.0",
        "Framework :: Plone :: 6.1",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    ],
    keywords="Python Plone CMS",
    author="Manabu TERADA",
    author_email="terada@cmscom.jp",
    url="https://github.com/collective/collective.vectorsearch",
    project_urls={
        "PyPI": "https://pypi.org/project/collective.vectorsearch/",
        "Source": "https://github.com/collective/collective.vectorsearch",
        "Tracker": "https://github.com/collective/collective.vectorsearch/issues",
        # 'Documentation': 'https://collective.vectorsearch.readthedocs.io/en/latest/',
    },
    license="GPL version 2",
    packages=find_packages("src", exclude=["ez_setup"]),
    namespace_packages=["collective"],
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "setuptools",
        # -*- Extra requirements: -*-
        "z3c.jbot",
        "plone.api>=1.8.4",
        "plone.app.dexterity",
        "plone.app.registry",
        "numpy",
        "fastembed>=0.2.0",
    ],
    extras_require={
        "test": [
            "plone.app.testing",
            # Plone KGS does not use this version, because it would break
            # Remove if your package shall be part of coredev.
            # plone_coredev tests as of 2016-04-01.
            "plone.testing>=5.0.0",
            "plone.app.contenttypes",
            "plone.app.robotframework[debug]",
        ],
        "gpu": [
            # GPU/CUDA support with PyTorch and Sentence Transformers
            "torch",
            "sentence_transformers",
            "transformers",
            "accelerate",
        ],
    },
    entry_points="""
    [z3c.autoinclude.plugin]
    target = plone
    [console_scripts]
    update_locale = collective.vectorsearch.locales.update:update_locale
    """,
)
