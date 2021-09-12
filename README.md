# Feature Subset Selection on SWAN_SF

The Space Weather ANalytics for Solar Flares (SWAN-SF) is a multivariate time series benchmark dataset that was recently constructed to serve as a testbed for solar flare forecasting models in the heliophysics community. SWAN-SF has 54 distinct features, including 24 quantitative features derived from active region photospheric magnetic field maps that describe previous flare activity.

Dataset used is SWAN-SF: Space Weather ANalytics for Solar Flares
You can find the dataset at "[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM]"

In this study, we thoroughly addressed the topic of quantifying the importance of these variables to the difficult task of flare forecasting . In order to obtain insight into the importance of the SWAN-SF features, we used a variety of methodologies in this study, which are summarized in below table. Filters and wrappers, supervised and unsupervised algorithms, univariate and multivariate approaches, and vectorized-based and Multivariate Time Series-based (MTS-based) strategies are all included in our list. We'll go over their implementations in this section.

![Table](/images/table.PNG)


For the preprocessing, feature selection, and assessment phases, we created an end-to-end pipeline.We methodologically compared the results of various FSS algorithms, both on multivariate time series and vectorized formats, and tested their correlation and reliability, to the extent possible, by using the selected features for flare forecasting on unseen data, in both univariate and multivariate fashions. Our research came to a close with a report on the best FSS methods in terms of top-k qualities, as well as an interpretation of the findings.


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
### Ranking

```
Run the code below in terminal to generate a ranking for each algorithm.

        >>> python mvts_fss_scs/__init__.py

```

### Evaluation

```

Run the code below in terminal to perform the evaluation.

        >>> python mvts_fss_scs/evaluation/evaluation1.py
        >>> python mvts_fss_scs/evaluation/evaluation2.py
        >>> python mvts_fss_scs/evaluation/evaluation3.py
        >>> python mvts_fss_scs/evaluation/evaluation4.py
        >>> python mvts_fss_scs/evaluation/evaluation5.py
        

```