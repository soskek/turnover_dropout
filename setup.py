from setuptools import setup, find_packages

setup(
    name="tod",
    version="0.1.0",
    license="MIT",
    packages=find_packages(include=['tod']),
    description='turnover dropout',
    install_requires=open('requirements.txt').read().splitlines(),
    author='Sosuke Kobayashi', author_email='sosk@preferred.jp',
    # url='https',
)
