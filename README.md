

# LAPRAS

[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][docs-url]



Lapras is developed to facilitate the dichotomy model development work.

## Usage
Lapras is designed to develop standard scorecard model. It contains the main steps as follows:  
1、Exploratory Data Analysis  
lapras.detect()  
lapras.quality()  
lapras.IV()  
lapras.VIF()  

2、Feature Selection  
lapras.select()  
lapras.stepwise()  

3、Binnings  
lapras.Combiner()  
lapras.WOETransformer()  
lapras.bin_stats()  
lapras.bin_plot()  

4、Modeling  
lapras.ScoreCard()  

5、Performance Measure  
lapras.perform()  
lapras.LIFT()  
lapras.score_plot()  
lapras.KS_bucket()  
lapras.PSI()  
lapras.KS()  
lapras.AUC()  

Also lapras provides a function which runs all the steps above automaticly:  
lapras.auto_model()  

For detailed usage, please refer to the wiki. Enjoy.  


## Install


via pip

```bash
pip install lapras --upgrade -i https://pypi.org/simple
```

via source code

```bash
python setup.py install
```




[pypi-image]: https://img.shields.io/badge/pypi-V0.0.15-%3Cgreen%3E
[pypi-url]: https://github.com/yhangang/lapras
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[docs-url]: https://github.com/yhangang/lapras

