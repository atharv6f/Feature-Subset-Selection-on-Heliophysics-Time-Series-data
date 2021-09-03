# Feature Subset Selection on SWAN_SF

### Setup Project

```
git clone https://username@bitbucket.org/gsudmlab/featuresubsetselection_on_swan-sf.git
cd featuresubsetselection_on_swan-sf
python -m venv venv
source ./venv/bin/activate
(venv) featuresubsetselection_on_swan-sf> pip3 install -r ./requirements.txt

```

### Project Structure

```
mvts_fss_scs
    |__ images
    |__ mvts_fss_scs
            |__ __init__.py
            |__ evaluation
                    |__ __init__.py
                    |__ metric.py
                    |__ plotter.py
                    |__ multivariate.py
                    |__ univariate.py
            |__ fss
                    |__ __init__.py
                    |__ base_fss.py
                    |__ clever
                    |__ corona
                    |__ csfs
                    |__ fcbf
                    |__ pie
                    |__ mrmr_relief
                    |__ rfe
            |__ preprocessing
                    |__ __init__.py
                    |__ imputer.py
                    |__ labeler.py
                    |__ normalizer.py
                    |__ sampler.py
                    |__ vectorizer.py
            |__ preprocessed_data_samples
            |__ sampled_data_samples

    |__ Results
    |__ CONSTANTS.py
    |__ README.md
    |__ requirements.txt
```
