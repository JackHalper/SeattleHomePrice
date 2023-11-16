
![Seattle Skyline](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Seattle%20Skyline%20Final.jpg)


# Redfin Real Estate Price Regressor for The City of Seattle

## Summary 
This model utilizes a two pronged approach; leveraging NLP on realtor descriptions with traditional housing appraisal features, and endeavors to provide independent real-estate investors and homeflippers in the City of Seattle with an accurate appraisal model while providing insight about which features are important in determining price in the Seattle market. The non-technical presentation can be found [here.](https://github.com/JackHalper/SeattleHomePrice/blob/main/Non-Technical%20Presentation.pdf)

## The Objective 
The aims of this project are two-fold; the first objective is to formulate as accurate of a housing price model as is feasible for the City of Seattle. The second is to leverage model-interpretability to deduce the features that predict housing values in the city market. Additionally, a secondary goal of this project is to utilize realtor descriptions and natural language processing to augment the performance of the housing model. 

## The Business Problem
This project aims to empower small-time value-add real estate investors in the city of Seattle. It attempts to do this in two distinct ways: 
1. It aims to provide an accurate appraisal model for homes based on their redfin listing information. 
2. It aims to provide information about what specific features cause some homes to sell for more so investors can have data-driven information through which to make value-add investment decisions.

## The Competition
[The Zestimate](https://www.zillow.com/z/zestimate/#/) by Zillow considers the following per FAQ's on the [Zillow Website](https://www.zillow.com/#/): 

"Home characteristics including square footage, location or the number of bathrooms.
On-market data such as listing price, description, comparable homes in the area and days on the market
Off-market data — tax assessments, prior sales and other publicly available records
Market trends, including seasonal changes in demand"

The Zestimate is a sophisticated Neural Network with access to unprecedented data on homes across the U.S. The Zestimate model incorporates 110,000,000 homes from across the United States. I will be evaluating my performance utilizing Zillow's performance with on-market listings and there eventual sales price. Given that this model requires descriptions to give predictions, this project will be evaluated on on-market listings as well. More on the Zestimate can be found [here](https://www.zillow.com/#/)

## The Data 
The Data was manually collected on November 31st, 2023 from the [Redfin](https://www.redfin.com/) Website. The redfin website provides an option to manually download CSV files with basic information about the property sale. Data were collected from all zip codes encompassing or partially encompassing the City of Seattle. Properties, not in the City of Seattle, were filtered out later in the analysis. Property information included in the download include: Beds, Baths, Price, Sale Date, Lot Size, Square Footage, Submarket Location, Longitude, Latitude, Sale Type, Year Built, Zipcode and More. The data cover the entirety of the city of Seattle and encompass all homesales registered on the website from the 12 months preceding the collection date. The homesales data is comprised of a variety of product types including; Single Family Detached Homes, Townhomes, Condos/Co-Ops, Multi-Family Properties, Vacant Land, Mobile Homes, Etc. For the purposes of this model and analysis, however, only SFD, TH's and Condos have been included due to other property types' relative infrequency in the Seattle City market. The data directly sourced from Redfin, however, lacked realtor descriptions. Therefore, the data collection process for this analysis included a [custom webscraping script](https://github.com/JackHalper/SeattleHomePrice/blob/main/WebScraper.ipynb) to pull descriptions from each properties unique redfin page. While the majority of homes sold in the City are listed on Redfin and include descriptions, some are not, and Redfin sources these description-less sales through MLS (Multiple Listing System). Properties that lacked descriptions were excluded for the purposes of the analysis and model. The data encompasses 6,308 Sales over the past 12 months in the City of Seattle. The Combined Final Redin Data used for the analysis and modeling can be found [here](https://github.com/JackHalper/SeattleHomePrice/blob/main/Redfin%20Data/Redfin_Combined.csv).


## Requirements
The Required Packages to run and reproduce this code can be found in [Requirements.txt](https://github.com/JackHalper/SeattleHomePrice/blob/main/Requirements.txt)


## How to run the Model on any Seattle Redfin Listing

Unfortunately, Data cannot be pulled on active listings in the same manner as sold properties. This is due to local regulations regarding Multiple Listing Systems (MLS) in the Seattle Area. Therefore, information about each property must be manually inputted into the tool. This tool runs off the best model(XGBRegressor) and the pickled model can be found [here.](https://github.com/JackHalper/SeattleHomePrice/blob/main/Pickled%20Models/XGBPipeline.pkl)

#### Instructions:

- Open [Redfin Housing Price Predictor Tool](https://github.com/JackHalper/SeattleHomePrice/blob/main/Redfin%20Housing%20Price%20Predictor.ipynb)
- Navigate to Tool Bar and Select Cell
- Click Run All 
- You will be prompted to input basic information about the property
  - Year Built
  - Property Type of which there are three options: (Case Sensitive)
    - Single Family Residential
    - Townhouse
    - Condo/Co-op
  - Zipcode
  - Beds
  - Baths
  - Location (This will be the nearest neighborhood on Google Maps; If there is an error try the nearest adjacent neighborhood; Remember to capitalize the location name)
  - Square Feet
  - Lot Size (in Sq. Ft.; 1 Acre = 43,560 Sq. Ft.)
  - Copy and Paste the Description
  - Enter the Listing Price for Comparison to the Prediction Output
  - Finally scroll to the bottom of the notebook to see the results!
    


## Exploratory Data Analysis
Disclaimer: the following section includes data only on properties qualifying for analysis and excludes some property types and descriptionless listings

#### Price: 

- Prices in the Dataset range from $202,500 - $11,620,000. 
- The Mean Home Price in the Set is $958K
- The Median Home Price in the Set is $820K
- The Dataset is skewed to the right

Price Histogram: 

![Price Histogram](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Distribution%20of%20Home%20Prices.png)

Price Boxplot Distribution by Property Type: 

![Price Boxplot](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Price%20by%20Property%20Type.png)

#### Square Footage: 

- Square Footages in the Dataset range from 268SF - 11,573 SF
- The Mean Home Square Footage in the Set is 1,729 SF
- The Median Home Square Footage in the Set is 1,568 SF
- The Dataset is skewed to the right

Square Footage Histogram: 

![SqFt. Histogram](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Distribution%20of%20Square%20Footages.png)

Square Footage Boxplot Distribution by Property Type: 

![SqFt. Boxplot](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Square%20Footage%20by%20Property%20Type.png)


#### Age

- Home Ages in the Dataset range from 0 - 133 years old
- The Mean Home Square Footage in the Set is 49 years of age
- The Median Home Square Footage in the Set is 44 years of age
- The Dataset is skewed to the right

Age Histogram: 

![Age Histogram](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Distribution%20of%20Home%20Ages.png)

Age Boxplot Distribution by Property Type: 

![SqFt. Boxplot](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Age%20by%20Property%20Type.png)

#### Beds 

- Number of Beds in the Dataset range from 0 - 8 Bedrooms
- The Mean Home in the Set has 2.8 Bedrooms
- The Median Home in the Set has 3 Bedrooms
- The Dataset is skewed slightly to left 


Bed Histogram: 

![Bed Histogram](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Beds%20Distribution.png)

Bed Boxplot Distribution by Property Type: 

![Bed Boxplot](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Beds%20by%20Property%20Type.png)


#### Baths

- Number of Beds in the Dataset range from 0 - 7 Bathrooms
- The Mean Home in the Set has 2.12 Bathrooms
- The Median Home in the Set has 2 Bathrooms
- The Dataset is skewed slightly to right 


Bath Histogram: 

![Bath Histogram](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Distribution%20of%20Property%20Baths.png)

Bath Boxplot Distribution by Property Type: 

![Bath Boxplot](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Baths%20by%20Property%20Type.png)


#### Natural Language Processing: Most Common Bigrams and Unigrams

![NLP Wordcloud](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Unigrams%20WordCloud%20Final.png)


## The Approach 

The approach taken, in this project, is basically to combine traditional housing features and NLP Bag-of-Words into ensemble regressors to acheive the most accurate result possible. The traditional numeric and categorical housing features were fed into a pipeline that either applied a Standard Scaler or One-Hot Encoding to prepare them for the regression models. The Description's were preprocessed, lemmatized, processed into a probable bigrams model and then vectorized into a simple bag of words array and fed into the model. 

![Approach Flowchart](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Approach%20FlowChart.png)

## The Metric 

Median Absolute Error Percentage & Mean Absolute Error Percentage 

- Median Absolute Error Percentage was selected as a primary metric due to it's preminence as the industry standard for evaluating home price models. Redfin Home Estimate and Zestimate pubilily assess their models with the median absolute error percentage. The percentage also provides easy interpretability and comparability across markets. An identical median absolute error would mean very different levels of accuracy if you were evaluating model performance in Huntsville, Alabama vis a vis Manhattan, NY. Therefore, a percentage metric allows for easy comprability of models in different housing markets context. 

- Mean Absolute error was selected as a secondary metric to ensure the model is not missing outliers by a huge margin 

## The Models 

#### Model #1: Decision Tree (No Hyperparameter Tuning) 
- The 1st model was a simple decision tree with no hyperparameter tuning just to provide a baseline model on which to evaluate future models.

Performance Metrics 
- This ensemble model acheived an R2 of approximately .68

- The Mean Absolute Error acheived was approximately $179K

- The Mean Absolute Percentage Error for this model is approximately 17.94% 

- The Median Absolute Percentage Error for this Model is Approximately 11.70%





Decision Tree Regressor Predicted vs. Actual Values:

![Decision Tree Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/AVP%20DTC.png)

Decision Tree Regressor Residuals vs. Actual Values:

![Decision Tree Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/RESIDUAL%20DTR.png)


#### Model #2: Random Forest (Hyperparameters Tuned)
- The 2nd model was produced through an exhaustive cross-validation gridsearch. 
- Final Hyperparameters:
  - Min_Samples_Leaf: 1
  - Max-Depth: None
  - Min_Samples_Split: 2  
  - n_estimators: 300

Performance Metrics 
- This ensemble model acheived an R2 of approximately .81

- The Mean Absolute Error acheived was approximately $139K

- The Mean Absolute Percentage Error for this model is approximately 14.46%. 

- The Median Absolute Percentage Error for this Model is Approximately 9.18%




Random Forest Regressor Predicted vs. Actual Values:

![Random Forest Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/AVP%20RFR.png)

Random Forest Regressor Residuals vs. Actual Values:

![Random Forest Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/RFR%20Residuals.png)



#### Model #3: XGBoost Regressor (Hyperparameters Tuned) (BEST MODEL)
- The 3rd model was produced through an exhaustive cross-validation gridsearch. 
- Final Hyperparameters:
  - .8 Proportion of Samples of Features Per Tree 
  - Max Depth of 10 
  - Minimum Child Weight of 4 
  - 500 total estimators in the ensemble model
  - .6 Proportion of samples of Features Per Tree. 

Performance Metrics 
- This ensemble model acheived an R2 of approximately .87 

- The Mean Absolute Error acheived was approximately 119k$

- The Mean Absolute Percentage Error for this model is approximately 12.64%. 

- The Median Absolute Percentage Error for this Model is Approximately 8.19%

  XGBoost Regressor Predicted vs. Actual Values:

  ![XGBoost Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/AVP%20XGB.png)

  XBBoost Regressor Residuals vs. Actual Values:

  ![XGBoost Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Residuals%20%20XGB.png)


#### Model #4: Support Vector Regressor
- The 4th model was produced through an exhaustive cross-validation gridsearch. 
- Final Hyperparameters:
  - C: 1000 (Low Regularization)
  - Gamma: 4
  - Epsilon: Scale
  - Kernal: Linear

Support Vector Regressor Predicted vs. Actual Values:

![Support Vector Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/AVP%20Stacked.png)

Support Vector Regressor Residuals vs. Actual Values:

![Support Vector Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Residuals%20SVR.png)


#### Model #5: Stacking Regressor

- The 5th model was produced through a stacked regressor combining the following models: 
- The Best Random Forest Regressor
- The Best XGBoost Regressor 
- The Meta Model is a Linear Regression Model


Performance Metrics 
- The Stacking Regressor Model performs in-line with the hyperparameter tuned XGBoost model.
- With a mean absolute error of 118k, it represents a slight but negligible improvement over the XGBoost
- The model has a mean absolute error percentage of just 12.55%, representing a miniscule improvement from the XBoost Regressor
- The model has a median absolute error percentage of just 8.25%, representing a miniscule regression from the XGBoost Regressor
- The Stacking Regressor, though performing in line with the XGBoost model, sacrifices model interpretability and therefore the XGBoost Regressor 
is selected as the superior model among all permutations tested.

Stacking Regressor Predicted vs. Actual Values:

![Stacking Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/AVP%20Stacked.png)

Stacking Regressor Residuals vs. Actual Values:

![Stacking Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Residuals%20Stakced.png)


## Best-Model (XGBRegressor) Feature Importances 

- Relevent Home Features: 
    - Walk-in-Pantry
    - Bonus Room 
    - Storage
    - Attached Garage
    - Mountains
    - Vaulted Ceilings
    - Adjacent to Restaurants & Shopping
    - Courtyard
- Most Relevent Home Variables
    - Bedrooms
    - Sale Type
    - Sold Date
    - Zipcode/Postal Code
    - Beds 
    - Year of Sale
    
- Conclusion: 
    - The utility of XGBoost feature importances is limited insofar as there is limited intepretability to how each feature specifically contributes to the target price variable. Certainly, we understand they are important to the prediction. However, it's less clear how they impact the target price variable. Therefore, these are important features but they must be contextualized with domain knowledge to provide real value


Top 30 Feature Importances

![Feature Importances](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Top%2030%20Feature%20Importances%20-%20Bigrams.png)

### Best Model - Comparison to Industry Benchmark

Zestimate (Seattle Metro) Median Absolute Error Percentage: 2.11%

Best Model Median Absolute Error Percentage: 8.19% 



### Overall Conclusion:
- The XGBoost Hyper-parameter tuned model performed the best while still providing some level of interpretability, whereas, the stacked regressor performed similarly provides no model intepretability
- The Project best model, as expected, significantly underperformed industry standard models with access to much larger reservoirs of information as well as historic pricing and 100's of millions of sales.
- The Project best model acheived a median absolute error percentage of 8.19% 
- Zillow's Zestimate, on the other hand, acheives significantly better performance of 2.11% in the Seattle Metro. 




### Next Steps and Possible Improvements
- A larger dataset, including more historic sales, would likely have increased the project best model performance.
- Additionally, Scraping additional features (eg. Garage Size, Finishings) from the Redfin website may have provided more numeric and categorical features to improve model performance.
- Topic Modeling, Such as Latent Dichlemet Analysis, could provide more salient NLP features and improved model performance.
- Expirimenting with NLP methods beyond a simple bigram Bag-of-Words approach might yield more precise predictions.


