from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas', 'scikit-learn', 'xgboost', 'matplotlib', 'google-cloud', 'google-cloud-storage',
                     'cloudml-hypertune']

setup(
    name='vertex_ai_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training package for Google Vertex AI'
)
