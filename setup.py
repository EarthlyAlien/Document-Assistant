from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="document-assistant",
    version="0.1.0",
    author="Document Assistant Contributors",
    author_email="example@example.com",
    description="A document processing system with RAG architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/document-assistant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain==0.0.267",
        "langchain-openai==0.0.2",
        "openai==1.3.5",
        "faiss-cpu==1.7.4",
        "sentence-transformers==2.2.2",
        "numpy==1.24.3",
        "python-dotenv==1.0.0",
        "pypdf==3.15.1",
        "streamlit==1.24.0",
    ],
) 