from setuptools import setup, find_packages

setup(
  name = 'blockwise-parallel-transformer',
  packages = find_packages(exclude=[]),
  version = '0.1.2',
  license='MIT',
  description = '32x Faster Attentionn',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/Blockwise-Parallel-Transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers',
    "Prompt Engineering"
  ],
  install_requires=[
    'jax',
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)