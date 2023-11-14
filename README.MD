
![Seattle Skyline](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Seattle%20Skyline%20Final.jpg)


# Redfin Real Estate Price Regressor for The City of Seattle

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

## The Metrics

Redfin and Zillow use the industry standard, Median Absolute Percentage Error (MEDAPE), to evaluate their respective pricing models: Zestimate and Redfin House Price Estimate. Given, MEDAPE's prominence as the Industry Standard barometer of performance, this project will primarily utilize this metric to evaluate model performance. 

Mean Absolute Precentange Error (MAPE) will be used as a secondary metric, because it's important to evaluate the model's performance in the context of mean to ensure it's not consistenly missing large outliers in the set. 

Zillow's Median Absolute Percentage Error in the Seattle Metro Area is approximately: 2.11%

## The Data 
The Data was manually collected on November 31st, 2023 from the [Redfin](https://www.redfin.com/) Website. The redfin website provides an option to manually download CSV files with basic information about the property sale. Data were collected from all zip codes encompassing or partially encompassing the City of Seattle. Properties, not in the City of Seattle, were filtered out later in the analysis. Property information included in the download include: Beds, Baths, Price, Sale Date, Lot Size, Square Footage, Submarket Location, Longitude, Latitude, Sale Type, Year Built, Zipcode and More. The data cover the entirety of the city of Seattle and encompass all homesales registered on the website from the 12 months preceding the collection date. The homesales data is comprised of a variety of product types including; Single Family Detached Homes, Townhomes, Condos/Co-Ops, Multi-Family Properties, Vacant Land, Mobile Homes, Etc. For the purposes of this model and analysis, however, only SFD, TH's and Condos have been included due to other property types' relative infrequency in the Seattle City market. The data directly sourced from Redfin, however, lacked realtor descriptions. Therefore, the data collection process for this analysis included a [custom webscraping script](https://github.com/JackHalper/SeattleHomePrice/blob/main/WebScraper.ipynb) to pull descriptions from each properties unique redfin page. While the majority of homes sold in the City are listed on Redfin and include descriptions, some are not, and Redfin sources these description-less sales through MLS (Multiple Listing System). Properties that lacked descriptions were excluded for the purposes of the analysis and model. The data encompasses 6,308 Sales over the past 12 months in the City of Seattle. 


## Requirements
The Required Packages to run and reproduce this code can be found in [Requirements.txt](https://github.com/JackHalper/SeattleHomePrice/blob/main/Requirements.txt)


## Initial Findings
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



## The Approach 

The approach taken, in this project, is basically, to combine traditional housing features and NLP Bag-of-Words into ensemble regressors to acheive the most accurate result possible. The traditional numeric and categorical housing features were fed into a pipeline that either applied a Standard Scaler or One-Hot Encoding to prepare them for the regression models. The Description's were preprocessed, lemmatized and count vectorized into a simple bag of words array and fed into the model.



## The Metric 

Median Absolute Error Percentage & Mean Absolute Error Percentage 

- Why???

## The Models 

#### Model #1: Decision Tree (No Hyperparameter Tuning) 
- The 1st model was a simple decision tree with no hyperparameter tuning just to provide a baseline model on which to evaluate future models.
This simple model acheived an R2 of approximately .65. The Mean Absolute Error acheived was approximately 183k$. The Mean Absolute Percentage Error is approximately 18.5%. 

Decision Tree Regressor Predicted vs. Actual Values:

![Decision Tree Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Act.%20Vs.%20Predicted%20DTC.png)

Decision Tree Regressor Residuals vs. Actual Values:

![Decision Tree Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Residuals%20DTC.png)


#### Model #2: Random Forest (Hyperparameters Tuned)
- The 2nd model was produced through an exhaustive cross-validation gridsearch. The Hyperparameters tuned include: Number of Estimators, Max Tree Depth, Minimum Sample Split, Minimum Samples Per Leaf. The best model from the CV search utilized the following final hyperparameters: No Max Depth, Minimum Sample Split of 2, Minimum Sample Per Leaf of 1, and 500 total trees in the ensemble model. This ensemble model acheived an R2 of approximately .82. The Mean Absolute Error acheived was approximately 137k$. The Mean Absolute Percentage Error is approximately 14.35%.

Random Forest Regressor Predicted vs. Actual Values:

![Random Forest Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/RFR%20Act.%20Vs.%20Predicted.png)

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
- This ensemble model acheived an R2 of approximately .88. 

- The Mean Absolute Error acheived was approximately 111k$. 

- The Mean Absolute Percentage Error for this model is approximately 11.64%. 

- The Median Absolute Percentage Error for this Model is Approximately 7.02%

  XGBoost Regressor Predicted vs. Actual Values:

  ![Decision Tree Regressor Predicted vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Act.%20Vs.%20Predicted%20XGB.png)

  XBBoost Regressor Residuals vs. Actual Values:

  ![Decision Tree Regressor Residuals vs. Actual Values](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/RFR%20Residuals.png)

  Top 20 Feature Importances 

  ![Feature Importances](https://github.com/JackHalper/SeattleHomePrice/blob/main/Photos/Top%2020%20Feature%20Importances.png)

   Honestly, not much can be deduced from this. Will attempt all grid searches and models on Bigrams tonight to see if changes model performance/interpretability.



## Best Model - Comparison to Industry Benchmark

Zestimate (Seattle Metro) Median Absolute Error Percentage: 2.11%

Best Model Median Absolute Error Percentage: 7.02% 


