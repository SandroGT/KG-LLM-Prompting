from setuptools import setup, find_packages


setup(
    name='llm-open-ie',
    version='0.1.0',
    description='Knowledge Graph Engineering through Iterative Zero-shot LLM Prompting',
    author='Salvatore Carta, Alessandro Giuliani, Marco Manolo Manca, Leonardo Piano, Alessandro Sebastian Podda,'
           ' Livio Pompianu, Sandro Gabriele Tiddia',
    author_email='salvatore@unica.it, alessandro.giuliani@unica.it, marcom.manca@unica.it, leonardo.piano@unica.it,'
                 ' sebastianpodda@unica.it, livio.pompianu@unica.it, sandrog.tiddia@unica.it',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'llm_open_ie.logger': ['*.ini']},
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=['openai', 'tiktoken']
)
