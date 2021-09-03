# Feature Subset Selection on SWAN_SF

### Setup Project

```
git clone https://username@bitbucket.org/gsudmlab/mvts_fss_scs.git
cd mvts_fss_scs
python -m venv venv
source ./venv/bin/activate
(venv) mvts_fss_scs> pip3 install -r ./requirements.txt

```

### Project Structure

```
mvts_fss_scs
    |__ _help
            |__ Dockerfile
            |__ jupyter_notebook_config.py
    |__ images
    |__ mvts_fss_scs
            |__ data
                    |__ partition 1
                        |__ FL
                        |__ NF
                    |__ partition 2
                        |__ FL
                        |__ NF
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

    |__ Results
    |__ CONSTANTS.py
    |__ README.md
    |__ README_internal.md
    |__ requirements.txt
```
