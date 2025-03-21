# 🌐 DSC288R Capstone Project - Winter 2024
**Project Group 01: Airbnb Price Prediction: Leveraging Machine Learning for Global Insights**

👥 **Team Members**:  
- Sneha Shah ([sns002@ucsd.edu](mailto:sns002@ucsd.edu))  
- Sean Deering ([sdeering@ucsd.edu](mailto:sdeering@ucsd.edu))  
- Mengkong Aun ([maun@ucsd.edu](mailto:maun@ucsd.edu))


# Abstract

## 🌍 Background
The rapid growth of rental platforms like Airbnb has significantly impacted housing markets worldwide, raising concerns about affordability and urban dynamics. Data science techniques such as ML can facilitate data-driven exploration of the extent of these impacts.<br>

## 🎯 Problem Definition
This project will predict Airbnb listing prices globally using ML regression models. Input consists of property characteristics, pricing, availability, reviews, host details, and geospatial data. Output will consist of price predictions, comparative analysis of model performance, and feature importance.<br>

## 💡 Motivation
The problem is well-suited for machine learning due to the dataset's richness, incorporating numerical, categorical, and textual features. The global scope introduces variance and complexity that warrants use of ML models capable of capturing regional variations.<br>

## 🔮 Literature Review
Previous approaches include use of AutoViz, CatBoost and SHAP[1] for global price prediction, development of web-based tools for host pricing assistance[2], and regional analyses focused on specific markets like Boston[3]. Our project differentiates itself by implementing multiple ML regression models on a global scale to evaluate their effectiveness in handling regional differences.

## ⚙️ Approach
We will implement Linear, Random Forest, Gradient Boosting, and K-Nearest Neighbors Regression. This approach will allow us to compare performance across different models and examine how each handle complex nonlinear relationships in the data. These models should complement each other well: Linear Regression will provide an interpretable baseline, Random Forest and Gradient Boosting will capture relationships between features, and KNN will capture local pricing patterns and neighborhood effects.<br>

## 🗂️ Dataset and Algorithm Details
The project will utilize Airbnb data sourced from Kaggle containing 89 features across nearly 500,000 listings, and appear to have been sourced from Inside Airbnb[4]. Given the complex nature of Airbnb pricing data and its non-linear relationships, we anticipate Gradient Boosting Regression to demonstrate superior performance, followed by Random Forest Regression.

## 🧬 Success Criteria
Model performance will be evaluated using standard regression metrics including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and cross-validation across different geographical regions. Feature importance analysis will inform the models' ability to capture key pricing factors.<br>

## 📚 References
1. dima806. (2023, September 20). *Airbnb global price predict AutoViz+CatBoost+SHAP*. Kaggle. [Link](https://www.kaggle.com/code/dima806/airbnb-global-price-predict-autoviz-catboost-shap)
2. *Predictive Price Modeling for Airbnb listings*. [Link](https://www.deepakkarkala.com/docs/articles/machine_learning/airbnb_price_modeling/about/index.html)
3. Wang, H. (2023). *Predicting Airbnb Listing Price with Different Models*. [DOI](https://doi.org/10.54097/hset.v47i.8169)
4. Inside Airbnb. (2024). *Get the Data*. [Link](https://insideairbnb.com/get-the-data/)


## 🚀 Getting Started

1) **Clone this repository and cd into it:**
   ```bash
   git clone https://github.com/deerings/Capstone.git
   cd Capstone

2) From the root directory of the repository, run 'make setup' to setup a virtual Python environment and install required dependencies. Type '.venv/bin/activate to activate the virtual environment.
   ```bash
   make setup
3) Type '.venv/bin/activate' to activate the virtual environment.
   ```bash
   source .venv/bin/activate

4) Next, run 'make setup-kaggle' and follow the directions to obtain your Kaggle API key. Make sure to save your key 'kaggle.json' to the '/Capstone/.kaggle/' folder, otherwise step 4 will not work. Alternatively, you can download the data directly from [kaggle](https://www.kaggle.com/datasets/joebeachcapital/airbnb/data) and save it to the /data folder of the repo.
   ```bash
   make setup-kaggle

5) After you've successfully completed Step 3, run 'make data'. This will pull the AirBNB dataset from Kaggle via the API and extract it into the /data folder.
   ```bash
   make data

6) open the '01 Data_Cleaning_EDA.ipynb' file found in /notebooks. This contains the code to run Data Cleaning, EDA, and Feature Engineering.

7) Export the cleaned data to the /data folder by running 'make clean-data'. This runs a separate .py script that is based on the '01 Data_Cleaning_EDA.ipynb' file. 
   ```bash
   make clean-data
8) open the '02 Models.ipynb' file found in /notebooks. This contains the code to run the models. Alternatively, if you just want to see the model output, you can run 'make models'.
    ```bash
   make models
9) To run the entire pipeline, type 'make pipeline' (to do this, you need to run 'make setup' and have successfully followed the instructions in 'make setup-kaggle' to obtain your Kaggle API key.
    ```bash
   make pipeline
10) running 'make cleanup' will remove all of the files generated by the code in this repository. This is not strictly necessary, as the /data folder has been added to .gitignore, however this will allow you to quickly remove the dataset/clean dataset if you wish.
   ```bash
   make cleanup
