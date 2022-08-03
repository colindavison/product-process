# Product/Process Patents

## Initial Files
- data/sic_list.csv: controls which focal industries will be predicted and any extra industries whose data will be used to help make the prediction for the focal industry.
- data/2040/2040_discern.zip: houses data on all publication claims in the industry, will be used by "code/classify.py" to make final predictions.

## code/classify_preamble.py
Please change root directory to your local directory and then alter any switches for changing output

## code/classify.py
Classifies publication claims in the "Grain Mill Products" industry (4-digit SIC 2040) as product or process innovations.  Starts with "data/2040/2040_discern_hand_classified.csv" which has publication claims that I hand classified.  Then, the code estimates sixty machine learning (ML) models.  The models are all combinations of three ML classifiers (multinomial naive bayes, complement naive bayes, and a passive
aggressive classifier), ten different text feature sets (i.e. the entire publication claims text and the 4-digit CPC code), and a dummy that indicates whether to keep all coefficients when predicting or drop coefficients that are below the mean in absolute value when predicting the outcome.  For each of these 60 models, I assess its quality using repeated K-fold cross validation where I choose K=5.  I choose the model that has the highest correlation coefficient between the vector of binary hand classifications and the vector of binary predictions. In a final step, I estimate seven more models where I add various features to the final selected model. If any of these extra seven models obtain a higher correlation coefficient between the truth and the prediction then the new model is chosen; otherwise I retain the previously chosen model.

### Output
- data/diagnostics.csv: documents diagnostic statistics (correlation coefficient, balanced accuracy, etc...) for the selected ML model
- data/2040/2040_discern_classified.csv: classifies all publication claims in industry 2040 as product or process innovations