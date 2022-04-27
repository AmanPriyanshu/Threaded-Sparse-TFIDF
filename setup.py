from setuptools import setup

setup(
    name='Threaded_Sparse_TFIDF',
    version='0.0',    
    description='Multithreading TF-IDF vectorization for similarity search using sparse matrices for computations.',
    url='https://github.com/AmanPriyanshu/Threaded-Sparse-TFIDF',
    author='Aman Priyanshu',
    author_email='amanpriyanshusms2001@gmail.com',
    license='BSD 2-clause',
    packages=['Threaded_Sparse_TFIDF'],
    install_requires=['tqdm>=4',
                      'nltk>=3',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 8',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)