import setuptools
with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    install_requires=required,
    name='alina_voice_assistant',
    version='0.1',
    scripts=['alina_voice_assistant'],
    author="Alice",
    description="Alina voice assistant training and recognition package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliceIllyuvieva/Alina-voice-assistant-",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
