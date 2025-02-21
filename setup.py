from setuptools import setup, find_packages


setup(
    name='DIABETES PREDICTION',  
    version='0.1.0', 
    author='M SHIVA RAMA KRISHNA REDDY',
    author_email='msrkreddy111@gmail.com',
    description='The main objective of the project is to use deep learning (DL) techniques to predict brain tumour types based on images',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/msrkreddy0928/Brain-Tumor-Prediction-DL-Project/tree/main',
    packages=find_packages(),  
    install_requires=[  
       'pandas',
        'numpy'
        'scikit-learn'
        'keras'
        'opencv'
        'flask'
        'scipy'
        'pymongo'
        'python-dotenv'

      
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10.16',
        'Operating System :: OS Independent',
    ],
    python_requires='3.10.16',  
)
