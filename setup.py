from setuptools import setup, find_packages

setup(
    name = 'clintrials',

    packages = find_packages(exclude=['gh-pages', 'doc', 'tutorials']),
    # packages = ['clintrials'],  # this must be the same as the name above

    version = '0.1.4',

    description = 'clintrials is a library of clinical trial designs and methods in Python',

    author = 'Kristian Brock',

    author_email = 'kristian.brock@gmail.com',

    url = 'https://github.com/brockk/clintrials',  # use the URL to the github repo

    download_url = 'https://github.com/brockk/clintrials/tarball/0.1.4',  # Should match a git tag

    keywords = ['clinical', 'trial', 'biostatistics', 'medical', 'statistics'],  # keywords

    classifiers = [

        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
)