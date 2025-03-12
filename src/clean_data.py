#!/usr/bin/env python
# coding: utf-8

# # 1. Data Cleaning Code

import pandas as pd

df = pd.read_csv('data/airbnb-listings.csv', sep=";", low_memory=False)

cols_to_drop = ["Space", "Neighborhood Overview", "Notes", "Transit", "Access", 
                "Interaction", "House Rules", "Host About", "Host Acceptance Rate",
                "Neighbourhood Group Cleansed", "Square Feet", "Weekly Price", 
                "Monthly Price", "Security Deposit", "Cleaning Fee", "License", "Jurisdiction Names", "Has Availability"]

df.drop(columns=cols_to_drop, inplace=True)


# #### 1a. Removal of ID and URL columns:
# We'll drop these columns as well since they won't add any predictive power to our models.

df.drop(columns=['Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url', 
                 'Host Thumbnail Url', 'Host Picture Url', 'ID','Listing Url', 'Scrape ID', 'Host ID',
                'Host URL'], inplace=True)


# #### 1b. Removal of other text based columns:
# We dropped Name, Description, and Summary since noise would be present if they are encoded by their mean price in Mean Target Encoding. Geolocation is also removed since it leads to redundancy and could lead to overfitting due to being highly correlated to Price. 

df = df.drop(columns=['Name', 'Description', 'Summary', 'Geolocation'])


# #### 2. Perform Data Imputation for columns with low to moderate missing values:
# We will deal with these columns by imputing median and modes or other appropriate values.

# In[20]:


# fill missing cols with appropriate values
categorical_fill_values = {
    "Host Response Time": "Unknown",
    "Host Neighbourhood": "Unknown",
    "State": df["State"].mode(dropna=True)[0] if "State" in df.columns else "Unknown",
    "Zipcode": df["Zipcode"].mode(dropna=True)[0] if "Zipcode" in df.columns else "00000",
    "Market": df["Market"].mode(dropna=True)[0] if "Market" in df.columns else "Unknown",
    #"Name": "Unnamed Listing",
    #"Summary": "No description",
    #"Description": "No description",
    "Property Type": df["Property Type"].mode(dropna=True)[0] if "Property Type" in df.columns else "Other",
    "Room Type": df["Room Type"].mode(dropna=True)[0] if "Room Type" in df.columns else "Unknown",
}

for col, value in categorical_fill_values.items():
    if col in df.columns:
        df[col] = df[col].fillna(value)

# specify the column names to fill with median values
numerical_fill_values = ["Host Response Rate", "Bathrooms", "Bedrooms", "Beds", "Price"]

for col in numerical_fill_values:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median(skipna=True))

# fill review-related numerical columns with median
review_cols = [
    "Review Scores Rating", "Review Scores Accuracy", "Review Scores Cleanliness",
    "Review Scores Checkin", "Review Scores Communication", "Review Scores Location",
    "Review Scores Value"
]

for col in review_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median(skipna=True))

# fill Reviews per Month with 0 since missing most likely means there weren't any reviews
if "Reviews per Month" in df.columns:
    df["Reviews per Month"] = df["Reviews per Month"].fillna(0)

# fill availability columns
availability_cols = ['Availability 30', 'Availability 60', 'Availability 90', 'Availability 365']
df[availability_cols] = df[availability_cols].fillna(0)

# fill remaining columns with appropriate values
df['Host Name'] = df['Host Name'].fillna("Unknown")
df['Host Since'] = df['Host Since'].fillna("Unknown")
df['Host Location'] = df['Host Location'].fillna("Unknown")
df['Street'] = df['Street'].fillna("Unknown")
df['Neighbourhood'] = df['Neighbourhood'].fillna("Unknown")
df['Neighbourhood Cleansed'] = df['Neighbourhood Cleansed'].fillna(df['Neighbourhood Cleansed'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['Smart Location'] = df['Smart Location'].fillna("Unknown")
df['Country Code'] = df['Country Code'].fillna(df['Country Code'].mode()[0])
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
df['Cancellation Policy'] = df['Cancellation Policy'].fillna(df['Cancellation Policy'].mode()[0])
df['Features'] = df['Features'].fillna("None")
df['Latitude'] = df['Latitude'].fillna(df['Latitude'].median())
df['Longitude'] = df['Longitude'].fillna(df['Longitude'].median())
df['Accommodates'] = df['Accommodates'].fillna(df['Accommodates'].median())
df['Guests Included'] = df['Guests Included'].fillna(df['Guests Included'].median())
df['Extra People'] = df['Extra People'].fillna(df['Extra People'].median())
df['Minimum Nights'] = df['Minimum Nights'].fillna(df['Minimum Nights'].median())
df['Maximum Nights'] = df['Maximum Nights'].fillna(df['Maximum Nights'].median())
df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
df['Host Listings Count'] = df['Host Listings Count'].fillna(0)
df['Host Total Listings Count'] = df['Host Total Listings Count'].fillna(0)
df['Calculated host listings count'] = df['Calculated host listings count'].fillna(0)
df['Last Scraped'] = df['Last Scraped'].fillna("2016-01-01")
df['Host Verifications'] = df['Host Verifications'].fillna("Unknown")  
df['Bed Type'] = df['Bed Type'].fillna(df['Bed Type'].mode()[0])
df['Amenities'] = df['Amenities'].fillna("No Amenities")
df['Calendar Updated'] = df['Calendar Updated'].fillna("Unknown")
df['Calendar last Scraped'] = df['Calendar last Scraped'].fillna("2016-01-01")
df['First Review'] = df['First Review'].fillna("2016-01-01")
df['Last Review'] = df['Last Review'].fillna("2016-01-01")
#df['Geolocation'] = df['Geolocation'].fillna("0, 0")

# confirm that all missing values have been successfully imputed
#print(df.isnull().sum().sum())

df.to_csv('data/airbnb-listings_cleaned.csv', index=False)
