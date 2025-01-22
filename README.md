# dsc288r_capstone_project
UCSD Master of Data Science DSC388R - Project Group 01: Airbnb Price Prediction: Leveraging Machine Learning for Global Insights - Winter 2024<br>
Sneha Shah (sns002@ucsd.edu), Sean Deering (sdeering@ucsd.edu), Mengkong Aun (maun@ucsd.edu)

# Background
The rapid growth of rental platforms like Airbnb has significantly impacted housing markets worldwide, raising concerns about affordability and urban dynamics. Data science techniques such as ML can facilitate data-driven exploration of the extent of these impacts.<br>

# Problem Definition
This project will predict Airbnb listing prices globally using ML regression models. Input consists of property characteristics, pricing, availability, reviews, host details, and geospatial data. Output will consist of price predictions, comparative analysis of model performance, and feature importance.<br>
# Motivation
The problem is well-suited for machine learning due to the dataset's richness, incorporating numerical, categorical, and textual features. The global scope introduces variance and complexity that warrants use of ML models capable of capturing regional variations.<br>

# Literature Review
Previous approaches include use of AutoViz, CatBoost and SHAP[1] for global price prediction, development of web-based tools for host pricing assistance[2], and regional analyses focused on specific markets like Boston[3]. Our project differentiates itself by implementing multiple ML regression models on a global scale to evaluate their effectiveness in handling regional differences.

# Approach
We will implement Linear, Random Forest, Gradient Boosting, and K-Nearest Neighbors Regression. This approach will allow us to compare performance across different models and examine how each handle complex nonlinear relationships in the data. These models should complement each other well: Linear Regression will provide an interpretable baseline, Random Forest and Gradient Boosting will capture relationships between features, and KNN will capture local pricing patterns and neighborhood effects.<br>

# Dataset and Algorithm Details
The project will utilize Airbnb data sourced from Kaggle containing 89 features across nearly 500,000 listings. If necessary, we will supplement the dataset with data from Inside Airbnb[4] or data augmentation. Given the complex nature of Airbnb pricing data and its non-linear relationships, we anticipate Gradient Boosting Regression to demonstrate superior performance, followed by Random Forest Regression.

# Success Criteria
Model performance will be evaluated using standard regression metrics including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and cross-validation across different geographical regions. Feature importance analysis will validate the models' ability to capture key pricing factors.<br>

# References

[1] dima806. (2023, September 20). Airbnb global price predict AutoViz+CatBoost+SHAP. Kaggle.com; Kaggle. https://www.kaggle.com/code/dima806/airbnb-global-price-predict-autoviz-catboost-shap
[2] Predictive Price Modeling for Airbnb listings. (n.d.). Www.deepakkarkala.com. https://www.deepakkarkala.com/docs/articles/machine_learning/airbnb_price_modeling/about/index.html
[3] Wang, H. (2023). Predicting Airbnb Listing Price with Different models. Highlights in Science Engineering and Technology, 47, 79–86. https://doi.org/10.54097/hset.v47i.8169
[4] Inside Airbnb. (2024). Get the Data. Insideairbnb.com. https://insideairbnb.com/get-the-data/

