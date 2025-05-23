from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='multimodal-rag',
    version='0.1.0',
    description='Multimodal RAG system with Pinecone and OpenAI Vision',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'openai>=1.0.0',
        'pinecone-client[grpc]>=3.0.0',
        'python-dotenv>=1.0.0',
        'PyPDF2>=3.0.0',
        'pdf2image>=1.16.3',
        'pytesseract>=0.3.10',
        'pandas>=2.0.0',
        'tqdm>=4.66.0',
        'python-multipart>=0.0.6',
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'python-magic>=0.4.27',
        'Pillow>=10.0.0',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'multimodal-rag=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
