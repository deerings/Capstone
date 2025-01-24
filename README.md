# dsc288r_capstone_project
UCSD Master of Data Science DSC288R - Project Group 01: Airbnb Price Prediction: Leveraging Machine Learning for Global Insights - Winter 2024<br>
Sneha Shah (sns002@ucsd.edu), Sean Deering (sdeering@ucsd.edu), Mengkong Aun (maun@ucsd.edu)
# Abstract

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

# Data Dictionary

| **Field**                                    | **Type**                  | **Calculated** | **Description**                                                                                                                                                                                                                                                                                                                                                      |
|---------------------------------------------|---------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id                                          | integer                   |                | Airbnb's unique identifier for the listing.                                                                                                                                                                                                                                                                                                                         |
| listing_url                                 | text                      | y              |                                                                                                                                                                                                                                                                                                                                                                      |
| scrape_id                                   | bigint                    | y              | Inside Airbnb "Scrape" this was part of.                                                                                                                                                                                                                                                                                                                             |
| last_scraped                                | datetime                  | y              | UTC. The date and time this listing was "scraped".                                                                                                                                                                                                                                                                                                                   |
| source                                      | text                      |                | One of "neighbourhood search" or "previous scrape". "neighbourhood search" means that the listing was found by searching the city, while "previous scrape" means that the listing was seen in another scrape performed in the last 65 days, and the listing was confirmed to be still available on the Airbnb site.                                                    |
| name                                        | text                      |                | Name of the listing.                                                                                                                                                                                                                                                                                                                                                |
| description                                 | text                      |                | Detailed description of the listing.                                                                                                                                                                                                                                                                                                                                |
| neighborhood_overview                       | text                      |                | Host's description of the neighbourhood.                                                                                                                                                                                                                                                                                                                            |
| picture_url                                 | text                      |                | URL to the Airbnb-hosted regular-sized image for the listing.                                                                                                                                                                                                                                                                                                       |
| host_id                                     | integer                   |                | Airbnb's unique identifier for the host/user.                                                                                                                                                                                                                                                                                                                       |
| host_url                                    | text                      | y              | The Airbnb page for the host.                                                                                                                                                                                                                                                                                                                                       |
| host_name                                   | text                      |                | Name of the host. Usually just the first name(s).                                                                                                                                                                                                                                                                                                                   |
| host_since                                  | date                      |                | The date the host/user was created. For hosts that are Airbnb guests, this could be the date they registered as a guest.                                                                                                                                                                                                                                             |
| host_location                               | text                      |                | The host's self-reported location.                                                                                                                                                                                                                                                                                                                                  |
| host_about                                  | text                      |                | Description about the host.                                                                                                                                                                                                                                                                                                                                         |
| host_response_time                          |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_response_rate                          |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_acceptance_rate                        |                           |                | The rate at which a host accepts booking requests.                                                                                                                                                                                                                                                                                                                  |
| host_is_superhost                           | boolean [t=true; f=false] |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_thumbnail_url                          | text                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_picture_url                            | text                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_neighbourhood                          | text                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_listings_count                         | text                      |                | The number of listings the host has (per Airbnb unknown calculations).                                                                                                                                                                                                                                                                                               |
| host_total_listings_count                   | text                      |                | The number of listings the host has (per Airbnb unknown calculations).                                                                                                                                                                                                                                                                                               |
| host_verifications                          |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_has_profile_pic                        | boolean [t=true; f=false] |                |                                                                                                                                                                                                                                                                                                                                                                      |
| host_identity_verified                      | boolean [t=true; f=false] |                |                                                                                                                                                                                                                                                                                                                                                                      |
| neighbourhood                               | text                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| neighbourhood_cleansed                      | text                      | y              | The neighbourhood as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.                                                                                                                                                                                                                                |
| neighbourhood_group_cleansed                | text                      | y              | The neighbourhood group as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.                                                                                                                                                                                                                         |
| latitude                                    | numeric                   |                | Uses the World Geodetic System (WGS84) projection for latitude and longitude.                                                                                                                                                                                                                                                                                        |
| longitude                                   | numeric                   |                | Uses the World Geodetic System (WGS84) projection for latitude and longitude.                                                                                                                                                                                                                                                                                        |
| property_type                               | text                      |                | Self-selected property type.                                                                                                                                                                                                                                                                                                                                        |
| room_type                                   | text                      |                | "[Entire home/apt | Private room | Shared room | Hotel]." All homes are grouped into these room types: Entire place, Private room, and Shared room.                                                                                                                                                                                                                          |
| accommodates                                | integer                   |                | The maximum capacity of the listing.                                                                                                                                                                                                                                                                                                                                |
| bathrooms                                   | numeric                   |                | The number of bathrooms in the listing.                                                                                                                                                                                                                                                                                                                             |
| bathrooms_text                              | string                    |                | "The number of bathrooms in the listing. On the Airbnb web-site, the bathrooms field has evolved from a number to a textual description. For older scrapes, bathrooms is used."                                                                                                                                                                                     |
| bedrooms                                    | integer                   |                | The number of bedrooms.                                                                                                                                                                                                                                                                                                                                             |
| beds                                        | integer                   |                | The number of bed(s).                                                                                                                                                                                                                                                                                                                                               |
| amenities                                   | json                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| price                                       | currency                  |                | "Daily price in local currency. NOTE: the $ sign is a technical artifact of the export, please ignore it."                                                                                                                                                                                                                                                           |
| minimum_nights                              | integer                   |                | Minimum number of night stay for the listing (calendar rules may be different).                                                                                                                                                                                                                                                                                      |
| maximum_nights                              | integer                   |                | Maximum number of night stay for the listing (calendar rules may be different).                                                                                                                                                                                                                                                                                      |
| minimum_minimum_nights                      | integer                   | y              | The smallest minimum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                              |
| maximum_minimum_nights                      | integer                   | y              | The largest minimum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                               |
| minimum_maximum_nights                      | integer                   | y              | The smallest maximum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                              |
| maximum_maximum_nights                      | integer                   | y              | The largest maximum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                               |
| minimum_nights_avg_ntm                      | numeric                   | y              | The average minimum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                                |
| maximum_nights_avg_ntm                      | numeric                   | y              | The average maximum_night value from the calendar (looking 365 nights in the future).                                                                                                                                                                                                                                                                                |
| calendar_updated                            | date                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| has_availability                            | boolean                   |                | [t=true; f=false].                                                                                                                                                                                                                                                                                                                                                  |
| availability_30                             | integer                   | y              | The availability of the listing 30 days in the future as determined by the calendar.                                                                                                                                                                                                                                                                                 |
| availability_60                             | integer                   | y              | The availability of the listing 60 days in the future as determined by the calendar.                                                                                                                                                                                                                                                                                 |
| availability_90                             | integer                   | y              | The availability of the listing 90 days in the future as determined by the calendar.                                                                                                                                                                                                                                                                                 |
| availability_365                            | integer                   | y              | The availability of the listing 365 days in the future as determined by the calendar.                                                                                                                                                                                                                                                                                |
| calendar_last_scraped                       | date                      |                |                                                                                                                                                                                                                                                                                                                                                                      |
| number_of_reviews                           | integer                   |                | The number of reviews the listing has.                                                                                                                                                                                                                                                                                                                               |
| number_of_reviews_ltm                       | integer                   | y              | The number of reviews the listing has (in the last 12 months).                                                                                                                                                                                                                                                                                                       |
| number_of_reviews_l30d                      | integer                   | y              | The number of reviews the listing has (in the last 30 days).                                                                                                                                                                                                                                                                                                        |
| first_review                                | date                      | y              | The date of the first/oldest review.                                                                                                                                                                                                                                                                                                                                 |
| last_review                                 | date                      | y              | The date of the last/newest review.                                                                                                                                                                                                                                                                                                                                  |
| review_scores_rating                        |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_accuracy                      |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_cleanliness                   |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_checkin                       |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_communication                 |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_location                      |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| review_scores_value                         |                           |                |                                                                                                                                                                                                                                                                                                                                                                      |
| license                                     | text                      |                | The license/permit/registration number.                                                                                                                                                                                                                                                                                                                              |
| instant_bookable                            | boolean                   |                | [t=true; f=false]. Whether the guest can automatically book the listing without the host requiring to accept their booking request. An indicator of a commercial listing.                                                                                                                                                                                             |
| calculated_host_listings_count              | integer                   | y              | The number of listings the host has in the current scrape, in the city/region geography.                                                                                                                                                                                                                                                                             |
| calculated_host_listings_count_entire_homes | integer                   | y              | The number of Entire home/apt listings the host has in the current scrape, in the city/region geography.                                                                                                                                                                                                                                                             |
| calculated_host_listings_count_private_rooms| integer                   | y              | The number of Private room listings the host has in the current scrape, in the city/region geography.                                                                                                                                                                                                                                                                |
| calculated_host_listings_count_shared_rooms | integer                   | y              | The number of Shared room listings the host has in the current scrape, in the city/region geography.                                                                                                                                                                                                                                                                |
| reviews_per_month                           | numeric                   | y              | The average number of reviews per month the listing has over the lifetime of the listing.                                                                                                                                                                                                                                                                            |


## UNDER CONSTRUCTION!
