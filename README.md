# LAPRAS

[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][docs-url]

Lapras is designed to make the model developing work easily and conveniently.
It contains these functions below in one key operation: data exploratory analysis, feature selection, feature binning,
data visualization, scorecard modeling(a logistic regression model with excellent interpretability), performance measure.

Let's get started.

## Usage

1.Exploratory Data Analysis
lapras.detect()
lapras.quality()
lapras.IV()
lapras.VIF()
lapras.PSI()

2.Feature Selection
lapras.select()
lapras.stepwise()

3.Binning
lapras.Combiner()
lapras.WOETransformer()
lapras.bin_stats()
lapras.bin_plot()

4.Modeling
lapras.ScoreCard()

5.Performance Measure
lapras.perform()
lapras.LIFT()
lapras.score_plot()
lapras.KS_bucket()
lapras.PPSI()
lapras.KS()
lapras.AUC()

6.One Key Auto Modeling
Lapras also provides a function which runs all the steps above automatically:
lapras.auto_model()

For more details, please refer to the wiki page. Enjoy.

## Install

via pip

```bash
pip install lapras --upgrade -i https://pypi.org/simple
```

via source code

```bash
python setup.py install
```

install_requires = [
'numpy >= 1.18.4',
'pandas >= 0.25.1, <=0.25.3',
'scipy >= 1.3.2',
'scikit-learn =0.22.2',
'seaborn >= 0.10.1',
'statsmodels >= 0.13.1',
'tensorflow >= 2.2.0, <=2.5.0',
'hyperopt >= 0.2.7',
'pickle >= 4.0',
]

[pypi-image]: https://img.shields.io/badge/pypi-V0.0.20-%3Cgreen%3E
[pypi-url]: https://github.com/yhangang/lapras
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[docs-url]: https://github.com/yhangang/lapras
