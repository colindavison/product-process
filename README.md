# Product/Process Patents

## Note
- Since Google Patents changes over time, the results may change slightly based on the day the data is accessed.  Google Patents was accessed for my economics job market paper on 5/18/2021.

## Outline
1. code/webscrape.py: webscrapes patent data from Google Patents
2. code/clean.py: cleans the scraped data
3. code/classify.py: uses my hand classifications to predict product/process classifications for all publication claims

## Initial Files
- data/diagnostics.csv: placeholder for diagnostic data
- data/patents.csv: lists all patents whose claims are to be classified
- data/sic_list.csv: controls which focal industries will be predicted and any extra industries whose data will be used to help make the prediction for the focal industry.
- data/hand_classified.csv: lists all hand classified publication claims

## code/webscrape.py
Scrapes all necessary data from Google Patents for a given industry.  The code can be readily modified to run in parallel on a compute cluster by using a list of industry codes.  
- Output: data/XXXX/XXXX_discern.csv (where XXXX is 4-digit SIC industry XXXX)

## code/clean.py
Cleans the raw data in XXXX_discern.csv and creates all variables needed as inputs for the machine learning (ML) models
- Output: data/XXXX/XXXX_discern.zip (where XXXX is 4-digit SIC industry XXXX)

## code/classify_preamble.py
Please change root directory to your local directory and then alter any switches for changing output before running "code/classify.py"

## code/classify.py
Classifies publication claims in all 4-digit SIC industries that are selected as product or process innovations.  The code estimates sixty ML models.  The models are all combinations of three ML classifiers (multinomial naive bayes, complement naive bayes, and a passive
aggressive classifier), ten different text feature sets (i.e. the entire publication claims text and the 4-digit CPC code), and a dummy that indicates whether to keep all coefficients when predicting or drop coefficients that are below the mean in absolute value when predicting the outcome.  For each of these 60 models, the code will assess its quality using repeated K-fold cross validation where K=5.  The code will then choose the model that has the highest correlation coefficient between the vector of binary hand classifications and the vector of binary predictions. In a final step, the code will estimate seven more models where various features are added to the final selected model. If any of these extra seven models obtain a higher correlation coefficient between the truth and the prediction then the new model is chosen; otherwise the previously chosen model is retained.
- Output
1. data/diagnostics.csv: documents diagnostic statistics (correlation coefficient, balanced accuracy, etc...) for the selected ML model
2. data/XXXX/XXXX_discern_classified.csv: classifies all publication claims in industry XXXX as product or process innovations