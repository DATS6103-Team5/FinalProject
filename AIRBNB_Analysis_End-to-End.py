# **************************Takes around 20-25 minutes to run the entire code**************************


#%%
# IMPORT LIBRARIES



# pip install -r /path/to/requirements.txt to install the required libraries.

import optuna
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier


import warnings
warnings.filterwarnings('ignore')




#%%
# SCRAPE DATA FROM AIRBNB WEBSITE



url = "http://insideairbnb.com/get-the-data/"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all links in the webpage
    links = soup.find_all("a")

    # List to store all the listings for USA.
    usa_listings = []

    # Iterate through the links
    for link in links:
        href = link.get("href")
        if href and ("united-states" in href) and href.endswith("listings.csv.gz"):
            usa_listings.append(href)
else:
    print("Failed to fetch webpage")


print(f"The data provided by AIRBNB contains litings across {len(usa_listings)} cities in USA") # Links for listings data for 34 cities in USA.
print(f"The data for the listings is stored at the links(Examples):\n{usa_listings[:5]}") # Links where the files are stored.





#%%


# DOWNLOAD THE DATA FOR ALL THE CITIES AND MERGE THEM TO CREATE A SINGLE DATAFRAME


# Load the data for cities.
city_data = []

for listing in tqdm(usa_listings):
  df = pd.read_csv(listing)
  df["link"] = listing
  city_data.append(df)

# Concatenate all the cities data to create a single dataframe.
usa_data = pd.concat(city_data)
display(usa_data.head())
print(f"Dataset Shape: {usa_data.shape}")



##########################################################################################################################################



#%%

# DATA CLEANING AND PREPROCESSING


#%%
## Part1 (Creating a trimmed dataset with the most useful columns)

# Working with a copy.
listings = usa_data.copy()
listings.reset_index(drop=True, inplace=True)
#listings.head()


# Create columns for City and State as they're not provided. 
listings["City"] = listings["link"].apply(lambda x: x.split('/')[5])
listings["State"] = listings["link"].apply(lambda x: x.split('/')[4])
display(listings[["host_location", "City", "State"]].head())
display(listings[["host_location", "City", "State"]].tail())


# Drop duplicate listings.
listings = listings.drop_duplicates(subset="id").reset_index(drop=True)


# Creat a starter dataset with the most useful listings columns.
starter_columns = ['id', 'host_id','host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_identity_verified', 'neighbourhood_cleansed',
                  'latitude', 'longitude', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews', 'number_of_reviews_ltm',
                  'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                  'review_scores_location', 'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 'reviews_per_month', 'City', 'State']

print(f"No of columns in the starter dataset: {len(starter_columns)}")


trimmed_listings = listings[starter_columns]
starter_listings_df = trimmed_listings.copy()



#%%

## Part2 (Dealing with data types)

# change host_response_time from categorical to numerical. Mapping  1, 2, 3, 4 - from fastest response to late response.
response_time_map = {"within an hour": 1,
                     "within a few hours": 2,
                     "within a day": 3,
                     "a few days or more": 4}

starter_listings_df["host_response_time"] = starter_listings_df["host_response_time"].map(response_time_map).astype("category")



# Convert numerical variables from strings to numerical.
starter_listings_df["host_response_rate"] = starter_listings_df["host_response_rate"].apply(lambda x: int(x[:-1]) if isinstance(x, str) else x)
starter_listings_df["host_acceptance_rate"] = starter_listings_df["host_acceptance_rate"].apply(lambda x: int(x[:-1]) if isinstance(x, str) else x)

# Convert binary variables to categories.
starter_listings_df["host_is_superhost"] = starter_listings_df["host_is_superhost"].apply(lambda x: 0 if x == 'f' else 1).astype("category")
starter_listings_df["host_identity_verified"] = starter_listings_df["host_identity_verified"].apply(lambda x: 0 if x == 'f' else 1).astype("category")
starter_listings_df["instant_bookable"] = starter_listings_df["instant_bookable"].apply(lambda x: 0 if x == 'f' else 1).astype("category")


# Extract number of bathrooms from bathrooms_text variable and remove the dollar sign from price variable.
starter_listings_df["bathrooms"] = starter_listings_df["bathrooms_text"].replace(['Half-bath', "Private half-bath", "Shared half-bath"], 0.5).apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x)
starter_listings_df["amenities"] = starter_listings_df["amenities"].apply(lambda x: len(eval(x)))
starter_listings_df["price"] = starter_listings_df["price"].apply(lambda x: float(''.join(x[1:].split(','))) if isinstance(x, str) else x)

# Convert the object variables to category.
starter_listings_df["neighbourhood_cleansed"] = starter_listings_df["neighbourhood_cleansed"].astype("category")
starter_listings_df["room_type"] = starter_listings_df["room_type"].astype("category")
starter_listings_df["City"] = starter_listings_df["City"].astype("category")
starter_listings_df["State"] = starter_listings_df["State"].astype("category")

# Drop the bathrooms_text and bedrooms (since bedrooms has very high number of missing values)
starter_listings_df.drop(["bathrooms_text", "bedrooms"], axis = 1, inplace = True)
starter_listings_df.info()

# Display summary of numerical and categorical variables.
num_cols = starter_listings_df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = starter_listings_df.select_dtypes(include=['category']).columns

print("Numerical and Categorical variables in the dataset:\n")
print(num_cols, cat_cols, len(num_cols), len(cat_cols))

print(starter_listings_df[num_cols].describe())
print(starter_listings_df[cat_cols].describe())


#%%


## Part3 (Handling Missing Values)


### Simple Imputation with Mean for numerical variables and Mode for categorical variables.


# Remove the listings with extreme price values.
starter_listings_df = starter_listings_df[starter_listings_df["price"] < 1200]

#display(simple_imputed_df.info())

simple_imputed_df = starter_listings_df.dropna(subset = ["price"]).copy()

simple_imputed_df[num_cols] = simple_imputed_df[num_cols].fillna(simple_imputed_df[num_cols].mean())
simple_imputed_df[cat_cols] = simple_imputed_df[cat_cols].fillna(simple_imputed_df[cat_cols].mode().iloc[0])

simple_imputed_df.reset_index(drop=True, inplace=True)
#Export the file to csv
#simple_imputed_df.to_csv('simple_imputed_df.csv', index=None)



### Imputation using Model


# Select the columns with missing values
num_cols_missing = ['host_response_rate', 'host_acceptance_rate','beds', 'review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness',
                    'review_scores_checkin', 'review_scores_communication', 'review_scores_location','review_scores_value', 'reviews_per_month', 'bathrooms']

cat_cols_missing = "host_response_time"

model_imputed_df = starter_listings_df.copy()

# Impute missing values for each numerical column using Histogram based Gradient Boosting Regressor
for col in tqdm(num_cols_missing):
    # Create a copy of the data with no missing values in the current column
    train_df = model_imputed_df[model_imputed_df[col].notnull()]
    X_train = train_df.drop([col, "neighbourhood_cleansed"], axis=1)
    X_train = pd.get_dummies(X_train, columns=['host_response_time', 'room_type', 'City', 'State'])
    y_train = train_df[col]

    # Build model
    hgbr_regressor = HistGradientBoostingRegressor()
    hgbr_regressor.fit(X_train, y_train)

    # Impute missing values
    X_test = model_imputed_df[model_imputed_df[col].isnull()].drop([col, "neighbourhood_cleansed"], axis=1)
    X_test = pd.get_dummies(X_test, columns=['host_response_time', 'room_type', 'City', 'State'])
    imputed_values = hgbr_regressor.predict(X_test)
    model_imputed_df.loc[model_imputed_df[col].isnull(), col] = imputed_values


# Impute missing values for host_response_time column using Histogram based Gradient Boosting Classifier
col = cat_cols_missing
train_df = model_imputed_df[model_imputed_df[col].notnull()]
X_train = train_df.drop([col, "neighbourhood_cleansed"], axis=1)
X_train = pd.get_dummies(X_train, columns=['room_type', 'City', 'State'])
y_train = train_df[col]

# Build model
hgbr_classifier = HistGradientBoostingClassifier()
hgbr_classifier.fit(X_train, y_train)

# Impute missing values
X_test = model_imputed_df[model_imputed_df[col].isnull()].drop([col, "neighbourhood_cleansed"], axis=1)
X_test = pd.get_dummies(X_test, columns=['room_type', 'City', 'State'])
imputed_values = hgbr_classifier.predict(X_test)
model_imputed_df.loc[model_imputed_df[col].isnull(), col] = imputed_values

model_imputed_df = model_imputed_df.dropna()
model_imputed_df["neighbourhood_cleansed"] = model_imputed_df["neighbourhood_cleansed"].fillna(model_imputed_df["neighbourhood_cleansed"].mode().iloc[0])
#model_imputed_df.info()

model_imputed_df.reset_index(drop=True, inplace=True)
#Export the file to csv
#simple_imputed_df.to_csv('model_imputed_df.csv', index=None)



##########################################################################################################################################



#%%

# EDA PART1 (EXPLORATORY DATA ANALYSIS)


"""
Review scores distribution
"""

plt.figure(figsize=(5,3))
plt.hist(simple_imputed_df['review_scores_rating'], bins=20)
plt.title('Review Score Distribution')
plt.xlabel('Review Scores Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()



#%%


"""
Compare room types in terms of price, rating, and number of listing.
"""


# Group data by 'room_type' and calculate average price, rating, total number of listing for each type of room
room_type_stats = simple_imputed_df.groupby('room_type').agg({
    'price': 'mean', 
    'review_scores_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'number_of_listings'}).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Price
sns.barplot(x='room_type', y='price', palette = "viridis", data=room_type_stats, ax=axes[0])
axes[0].set_title('Average Price by Room Type')

# Rating
sns.barplot(x='room_type', y='review_scores_rating', palette = "viridis", data=room_type_stats, ax=axes[1])
axes[1].set_ylim(4.6, 5)
axes[1].set_title('Average Rating by Room Type')

# Number of listings
sns.barplot(x='room_type', y='number_of_listings', palette = "viridis", data=room_type_stats, ax=axes[2])
axes[2].set_title('Number of Listings by Room Type')

# Rotate x-axis label
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

plt.tight_layout()
plt.show()



#%%


"""
Compare city in terms of price, rating, and number of listings.
"""


# Group data by 'City' and calculate average price, rating, total number of listing for each city
city_stats = simple_imputed_df.groupby('City').agg({
    'price': 'mean', 
    'review_scores_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'number_of_listings'}).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Price
sns.barplot(x='City', y='price', palette = "coolwarm", data=city_stats, ax=axes[0])
axes[0].set_title('Average Price by City')

# Rating
sns.barplot(x='City', y='review_scores_rating', palette = "coolwarm", data=city_stats, ax=axes[1])
axes[1].set_ylim(4.5, 5)
axes[1].set_title('Average Rating by City')

# Number of listings
sns.barplot(x='City', y='number_of_listings', palette = "coolwarm", data=city_stats, ax=axes[2])
axes[2].set_title('Number of Listings by City')

# Rotate x-axis label
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

plt.tight_layout()
plt.show()


#%%


"""
Do superhosts tend to have higher review scores, more bookings compared to regular hosts?
"""


fig, ax = plt.subplots(1, 2, figsize = (12, 5))
# Review scores vs host status
means = simple_imputed_df.groupby(["host_is_superhost"])["review_scores_rating"].median()
means.plot(kind = "bar", color = ['skyblue', 'lightgreen'], ax = ax[0])


# Title and labels
ax[0].set_title('Review scores vs Host Status')
ax[0].set_xlabel('Host Status', fontsize=14)
ax[0].set_ylabel('Review Scores (Median)', fontsize=14)
ax[0].set_xticks(ticks=[0, 1], labels=['Regular Host', 'Super Host'], rotation = 0);

for p in ax[0].patches:
    ax[0].annotate('{:.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')


# Number of reviews vs different types of hosts
means = simple_imputed_df.groupby(["host_is_superhost"])["number_of_reviews"].median()
means.plot(kind = "bar", color = ['#ccff99', 'pink'], ax = ax[1])

# Title and labels
ax[1].set_title('No of review vs Host Status')
ax[1].set_xlabel('Host Status', fontsize=14)
ax[1].set_ylabel('Number of reviews (Median)', fontsize=14)
ax[1].set_xticks(ticks=[0, 1], labels=['Regular Host', 'Super Host'], rotation = 0);

for p in ax[1].patches:
    ax[1].annotate('{:.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
     


#%%%


"""
How does managing multiple listings affect the review scores
"""


single_df = simple_imputed_df.query("calculated_host_listings_count == 1").reset_index()["review_scores_rating"]
group_df = simple_imputed_df.query("calculated_host_listings_count > 1").groupby('host_id')['review_scores_rating'].median().reset_index()["review_scores_rating"]

df = pd.DataFrame({"scores": single_df.tolist() + group_df.tolist(),
              "type": ["single"]*single_df.shape[0]+ ["multi"]*group_df.shape[0]})

plt.figure(figsize=(8, 4))
plt.ylim(3, 5)
sns.boxplot(x='type', y='scores', data=df, palette = ["#FF5733", "#E6E6FA"])
plt.xlabel('Host Listings count')
plt.ylabel('Review Score')
plt.title('Box Plot of Review Scores for Single vs. Multiple Listings Hosts')
plt.xticks(ticks=[0, 1], labels=['Hosts with Single Listing', 'Hosts with Multiple Listings'])
plt.yticks([3, 4, 5])
plt.show()



#%%


"""
Does host_response_time has an impact on the Review Scores?
"""


plt.figure(figsize=(8, 4))
plt.ylim(1, 5)
sns.boxplot(x='host_response_time', y='review_scores_rating', data=simple_imputed_df, palette = "Paired")
plt.title('Impact of Host Response Time on Ratings')
plt.xlabel('Host Response Time')
plt.ylabel('Review Scores Rating')
plt.yticks(range(1, 6))
plt.show()



#%%


"""
Compare states in terms of price, rating, and number of listings.
"""


# https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.

abbreviation_to_name = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
    "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming",
    "DC": "District of Columbia", "AS": "American Samoa", "GU": "Guam GU", "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico PR", "VI": "U.S. Virgin Islands",
}


# Group data by 'State' and calculate average price, rating, total number of listing for each city
state_stats = simple_imputed_df.groupby('State').agg({
    'price': 'mean',
    'review_scores_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'number_of_listings'}).reset_index()

state_stats["State"] = state_stats["State"].apply(lambda x: x.upper()).map(abbreviation_to_name)

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Price
sns.barplot(x='State', y='price', data=state_stats, ax=axes[0], palette  = sns.color_palette("pastel"))
axes[0].set_title('Average Price by State')


for i, bar in enumerate(axes[0].containers):
    neighborhood = f"{state_stats.iloc[i]['price']:.2f}"
    axes[0].bar_label(bar, labels=[neighborhood], rotation=0, fontsize=10, padding=5)

# Review Scores Rating
sns.barplot(x='State', y='review_scores_rating', data=state_stats, ax=axes[1], palette = sns.color_palette("pastel"))
axes[1].set_ylim(4.5, 5)
axes[1].set_title('Average Rating by State')

for i, bar in enumerate(axes[1].containers):
    neighborhood = f"{state_stats.iloc[i]['review_scores_rating']:.2f}"
    axes[1].bar_label(bar, labels=[neighborhood], rotation=0, fontsize=10, padding=5)

# Number of listings
sns.barplot(x='State', y='number_of_listings', data=state_stats, ax=axes[2], palette = sns.color_palette("pastel"))
axes[2].set_title('Number of Listings by State')

for i, bar in enumerate(axes[2].containers):
    neighborhood = state_stats.iloc[i]['number_of_listings']
    axes[2].bar_label(bar, labels=[neighborhood], rotation=0, fontsize=10, padding=5)

axes[2].set_xticklabels(labels = state_stats["State"], rotation = 45)

plt.tight_layout()
plt.show()



#%%


"""
Neighbourhoods with highest no of listings in each City.
"""


city_neighborhood_counts = simple_imputed_df.groupby(['City', 'neighbourhood_cleansed']).size().reset_index(name='count')
top_neighborhoods = city_neighborhood_counts.sort_values(['City', 'count'], ascending=[True, False]).groupby('City').head(1)

plt.figure(figsize=(21, 7))
bar_ax = sns.barplot(x="City", y="count", data=top_neighborhoods, palette="viridis")

# Add neighborhood names and their count.
for i, bar in enumerate(bar_ax.containers):
    neighborhood = str(top_neighborhoods.iloc[i]['neighbourhood_cleansed']) + f" (Count: {top_neighborhoods.iloc[i]['count']})"
    if top_neighborhoods.iloc[i]["count"] > 10000:
      rotation = 0
    else:
      rotation = 80
    bar_ax.bar_label(bar, labels=[neighborhood], rotation=rotation, fontsize=10, padding=5)


plt.xlabel("City")
plt.ylabel("Number of listings")
plt.xticks(ticks=top_neighborhoods["City"], fontsize=12, rotation=60)
plt.title("Top neighborhood in each city")
plt.show()



#%%


"""
Costliest neighbourhood in each city.
"""


# Calculate the average price for each city and neighborhood combination
avg_prices = simple_imputed_df.groupby(['City', 'neighbourhood_cleansed'])['price'].median().reset_index()

# Merge the average prices with the top neighborhood counts
top_neighborhoods = city_neighborhood_counts.merge(avg_prices, on=['City', 'neighbourhood_cleansed'])

# Sort by city and average price (descending)
top_neighborhoods = top_neighborhoods.sort_values(['City', 'price'], ascending=[True, False]).groupby('City').head(1)

plt.figure(figsize=(21, 8))
bar_ax = sns.barplot(x="City", y="price", data=top_neighborhoods, palette="magma")

# Add neighborhood names and the median price
for i, bar in enumerate(bar_ax.containers):

    if top_neighborhoods.iloc[i]["price"] > 750:
      rotation = 0
    else:
      rotation = 80

    neighborhood = str(top_neighborhoods.iloc[i]['neighbourhood_cleansed']) + f" (${top_neighborhoods.iloc[i]['price']:.2f})"
    bar_ax.bar_label(bar, labels=[neighborhood], rotation=rotation, fontsize=9)


plt.xlabel("City")
plt.ylabel("Median Price")
plt.xticks(ticks=top_neighborhoods["City"], fontsize=12, rotation=60)
plt.title("Costliest Neighborhood in Each City")
plt.show()



#%%


"""
Neighbourhoods with more than 1000 listings (Popular Neighbourhoods)
"""


# Get the count of each neighborhood
neighborhood_counts = simple_imputed_df["neighbourhood_cleansed"].value_counts()

# Filter out neighborhoods with less than 1000 listings
common_neighborhoods = neighborhood_counts[neighborhood_counts > 1000]

common_neighborhoods = common_neighborhoods[common_neighborhoods.index.str.contains('Areas|Center|District|town|Side|Village') == False].to_frame().reset_index().rename(columns = {"index": "neighbourhood_cleansed", "neighbourhood_cleansed": "count"})

bar_ax = common_neighborhoods.plot(kind = "bar", colormap="Accent", figsize = (15, 6))
plt.title("Neighbourhoods with more than 1000 listings (Popular Neighbourhoods)")
plt.xlabel("Neighbourhood", fontsize=14)
plt.ylabel("Number of listings")
plt.xticks(ticks = range(common_neighborhoods.shape[0]), labels=common_neighborhoods["neighbourhood_cleansed"], fontsize=12, rotation=60)

for i in range(common_neighborhoods.shape[0]):
    plt.text(i, common_neighborhoods["count"][i] + 50, common_neighborhoods["count"][i], ha = 'center')


plt.show()


##########################################################################################################################################



#%%

# EDA PART2 (Analysis of Price)


"""
Price distribution
"""


plt.figure(figsize=(5,3))
plt.hist(simple_imputed_df['price'], bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


#%%


"""
Use scatter plot to compare price with numerical variables
"""


numerical = num_cols

plt.figure(figsize=(20, 20))   
for i in range(len(numerical)):
    plt.subplot(4, 5, i+1)
    plt.scatter(simple_imputed_df[numerical[i]], simple_imputed_df['price'])
    plt.xlabel(numerical[i])
    plt.ylabel('price')
    plt.title(f'{numerical[i]} vs price')
    if i == 19:
        break
plt.tight_layout()
plt.show()


# %%


"""
Use line chart to compare average price with numerical variables
"""


plt.figure(figsize=(20, 20))
for i in range(len(numerical)):
    plt.subplot(4, 5, i+1)
    simple_imputed_df.groupby(numerical[i])['price'].mean().plot()
    plt.xlabel(numerical[i])
    plt.ylabel('price')
    plt.title(f'{numerical[i]} vs price')

    if i == 19:
        break
plt.tight_layout()
plt.show()
# %%


"""
Use box plot to compare price with categorical variables
"""

categorical = ["host_is_superhost", "host_identity_verified", "room_type", "instant_bookable", "City", "State"]
for i in range(len(categorical)):
    if categorical[i] == "City" or categorical[i] == "State":
        plt.figure(figsize=(12, 6))
        plt.xticks(rotation=75)
    sns.boxplot(x=categorical[i], y='price', data=simple_imputed_df, palette = "cool", showfliers=False)
    plt.xlabel(categorical[i])
    plt.ylabel('price')
    plt.title(f'{categorical[i]} vs price')
    plt.show()



# %%

"""
Plot a heatmap of correlation matrix of numerical variables
"""

plt.figure(figsize=(20, 20))
sns.heatmap(simple_imputed_df[numerical].corr(), annot=True, cmap='coolwarm')
plt.show()

#%%

"""
Scatter plot of price vs. review per month and number of reviews
"""

# Filter data for listings with reviews in the last 6 months
recent_reviews = simple_imputed_df[simple_imputed_df['number_of_reviews_ltm'] > 0]

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='reviews_per_month', y='price', hue='number_of_reviews_ltm', data=recent_reviews, palette='coolwarm')
plt.title('Price vs. Reviews per Month')
plt.xlabel('Reviews per Month')
plt.ylabel('Price')
plt.legend(title='Number of Reviews (Last 12 Months)')
plt.show()



# END OF EDA
##########################################################################################################################################





#%%


# MODELING


#%% 

# Import the data.

sidf = pd.read_csv("https://raw.githubusercontent.com/DATS6103-Team5/FinalProject/main/data/simple_imputed_df.csv")
midf = pd.read_csv("https://raw.githubusercontent.com/DATS6103-Team5/FinalProject/main/data/model_imputed_df.csv")


display(sidf.head())
display(midf.head())


model_cols = ["host_response_rate", "host_acceptance_rate", "latitude", "longitude", "accommodates", "beds", "number_of_reviews", "number_of_reviews_ltm",
              "review_scores_rating", "reviews_per_month", "bathrooms", "price"]

model_sidf = sidf[model_cols]
display(model_sidf.info())


#%%

# Prepare the Data for Modeling

X = model_sidf.drop(["price"], axis = 1)
y = model_sidf["price"]

print("Shapes:", X.shape, y.shape)


# Split the data into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
print("Shapes after splitting:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#%%


"""
Creating Baseline Models
"""


# Creating a function to display the train and test metrics.

def display_results(model, y_train, train_preds, y_test, test_preds, exp):

  train_mae = mean_absolute_error(y_train, train_preds)
  train_rmse = mean_squared_error(y_train, train_preds)**0.5

  test_mae = mean_absolute_error(y_test, test_preds)
  test_rmse = mean_squared_error(y_test, test_preds)**0.5

  model_name = model.__class__.__name__

  print(f"Results for {model_name}:")
  print(f"Experiment: {exp}")
  print(f"Mean Absolute Error for Training data: {train_mae}")
  print(f"Root Mean Squared Error for Training data: {train_rmse}")

  print(f"Mean Absolute Error for Test data: {test_mae}")
  print(f"Root Mean Squared Error for Test data: {test_rmse}")

  return {"Model": model_name,
          "exp": exp,
          "Train MAE": train_mae,
          "Train RMSE": train_rmse,
          "Test MAE": test_mae,
          "Test RMSE": test_rmse}



# Linear Regression


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
lr_train_preds = linear_regressor.predict(X_train)
lr_test_preds = linear_regressor.predict(X_test)

lr_results = display_results(linear_regressor, y_train, lr_train_preds, y_test, lr_test_preds, exp = "Linear Regression")


# From the train and test results we can see that the model is underfitting. 
# So let's increase the complexity of the models.

# %%

# Decision Tree Regressor

dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)
dtr_train_preds = dtr_model.predict(X_train)
dtr_test_preds = dtr_model.predict(X_test)

dtr_results = display_results(dtr_model, y_train, dtr_train_preds, y_test, dtr_test_preds, exp = "Decision Tree Regression")


# The training results are great but the test results shows that the model actually overfitted. 
# Let's see if we can regularize the model.

# %%

# Decision Tree Regressor with hand picked parameter Tuning

dtr_model = DecisionTreeRegressor(max_depth = 15, min_samples_split = 40, min_samples_leaf = 10, min_impurity_decrease=0.0001)
dtr_model.fit(X_train, y_train)
dtr_train_preds = dtr_model.predict(X_train)
dtr_test_preds = dtr_model.predict(X_test)

dtr_results_tuned = display_results(dtr_model, y_train, dtr_train_preds, y_test, dtr_test_preds, exp = "Decision Tree Regression Tuned")

# Looks like with a little bit of tuning (Just hand picked) the important parameters improved the model 
# significantly. Restricting the model brought the balance between the training and test performance.

# # Let's see if we can keep on increasing the complexity of models and see how far we can go.


# %%

# Random Forest Regressor

rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, y_train)
rfr_train_preds = rfr_model.predict(X_train)
rfr_test_preds = rfr_model.predict(X_test)

rfr_results = display_results(rfr_model, y_train, rfr_train_preds, y_test, rfr_test_preds, exp = "Random Forest Regressor")

# Again we can see that the model is overfitting a bit. 
# So we need to tune the model parameters to balance the performance between the train and test.

# %%

# Random Forest Regressor with hand picked parameter Tuning

rfr_model = RandomForestRegressor(n_estimators=100, max_depth = 15, min_samples_split = 40, min_samples_leaf = 10)
rfr_model.fit(X_train, y_train)
rfr_train_preds = rfr_model.predict(X_train)
rfr_test_preds = rfr_model.predict(X_test)

rfr_results_tuned = display_results(rfr_model, y_train, rfr_train_preds, y_test, rfr_test_preds, exp = "Random Forest Regressor Tuned")

#%%

# XGBoost Regressor

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_train_preds = xgb_model.predict(X_train)
xgb_test_preds = xgb_model.predict(X_test)

xgb_results = display_results(xgb_model, y_train, xgb_train_preds, y_test, xgb_test_preds, exp = "XGBoost")

# It took a very few seconds for training XGB model compared to RandomForest model. 
# And also the performance on test set (generalization) is better than the previous models.

# %%

# LightGBM Regressor

lgbm_model = lgbm.LGBMRegressor(num_iterations = 1000, verbose = -1)
lgbm_model.fit(X_train, y_train)
lgbm_train_preds = lgbm_model.predict(X_train)
lgbm_test_preds = lgbm_model.predict(X_test)

lgbm_results = display_results(lgbm_model, y_train, lgbm_train_preds, y_test, lgbm_test_preds, exp = "LightGBM")

# Light GBM outperformed all other baseline models. It reduces the errors and also ensured generalization.


#%%

"""
Let's compare the results of all the baseline models.
"""

results_sidf = pd.DataFrame([lr_results, dtr_results, dtr_results_tuned, rfr_results, rfr_results_tuned, xgb_results, lgbm_results])


melted_sidf = results_sidf[["exp", "Train MAE", "Test MAE"]].melt(id_vars=["exp"], var_name="Type", value_name="MAE")

plt.figure(figsize=(12, 5))
bar_containers = sns.barplot(x="exp", y="MAE", data=melted_sidf, hue="Type")
plt.xticks(rotation=45)


all_bars = []
for bar_container in bar_containers.containers:
    all_bars.extend(bar_container.get_children())

labels = [f"{bar.get_height():.2f}" for bar in all_bars]


for bar, label in zip(all_bars, labels):
    bar_height = bar.get_height()
    x_coord = bar.get_x() + bar.get_width() / 2
    y_coord = bar_height + bar.get_y() + 0.01
    plt.text(x_coord, y_coord, label, ha='center', va='bottom')

plt.title("Model performance on Train and Test sets")
plt.xticks(ticks=melted_sidf["exp"], rotation=45)

plt.show()



# %%


# Using Modeling and Preprocessing techniques to improve the performance of models.

# Modelling Part 2 (Improving Performance)

#%%

"""
Can model based imputation improve the price prediction?
"""

model_midf = midf[model_cols]

X = model_midf.drop(["price"], axis = 1)
y = model_midf["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%

# Creating a basic function that can be used for training and evaluating any model (To remove redundancy).

def train_and_eval(model_obj, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
    
  # Train
  model = model_obj
  model.fit(X_train, y_train)

  # Predictions
  train_preds = model.predict(X_train)
  test_preds = model.predict(X_test)

  # Evaluation
  results = display_results(model, y_train, train_preds, y_test, test_preds, exp = model.__class__.__name__)

  return results

#%%

# Repeating the basline modeling with model_imputed_df.

lr_model = LinearRegression()
dtr_model = DecisionTreeRegressor()
dtr_model_tuned = DecisionTreeRegressor(max_depth = 15, min_samples_split = 40, min_samples_leaf = 10, min_impurity_decrease=0.0001)
rfr_model = RandomForestRegressor()
rfr_model_tuned = RandomForestRegressor(n_estimators=100, max_depth = 15, min_samples_split = 40, min_samples_leaf = 10)
xgb_model = xgb.XGBRegressor()
lgbm_model = lgbm.LGBMRegressor(num_iterations = 1000, verbose = -1)



lr_results_midf = train_and_eval(lr_model)
dtr_results_midf = train_and_eval(dtr_model)
dtr_results_tuned_midf = train_and_eval(dtr_model_tuned)
rfr_results_midf = train_and_eval(rfr_model)
rfr_results_tuned_midf = train_and_eval(rfr_model_tuned)
xgb_results_midf = train_and_eval(xgb_model)
lgbm_results_midf = train_and_eval(lgbm_model)


#%%

"""
Comparing the results of model_imputed_df with simple_imputed_df.
"""


results_midf = pd.DataFrame([lr_results_midf, dtr_results_tuned_midf, rfr_results_tuned_midf, xgb_results_midf, lgbm_results_midf])

results_sidf2 = pd.DataFrame([lr_results, dtr_results_tuned, rfr_results_tuned, xgb_results, lgbm_results])
results_sidf2["df"] = "Simple Imputation"
results_midf["df"] = "Model Imputation"
result_cat = pd.concat([results_sidf2[["Model", "Train MAE", "Test MAE", "df"]], results_midf[["Model", "Train MAE", "Test MAE", "df"]]])


df = result_cat.copy()

# Extract data from the dataframe
models = df['Model'].values
train_simple = df[df['df'] == 'Simple Imputation']['Train MAE'].values
test_simple = df[df['df'] == 'Simple Imputation']['Test MAE'].values
train_model = df[df['df'] == 'Model Imputation']['Train MAE'].values
test_model = df[df['df'] == 'Model Imputation']['Test MAE'].values

# Set up the x-axis positions for the groups
x = np.arange(len(train_simple)) * 3
width = 0.65  # Adjust the bar width as desired

# Create the figure and axis
fig, ax = plt.subplots(figsize=(13, 6))

# Plot the grouped bars
train_simple_bars = ax.bar(x - width, train_simple, width, label='Train Simple Imputation')
test_simple_bars = ax.bar(x, test_simple, width, label='Test Simple Imputation')
train_model_bars = ax.bar(x + width, train_model, width, label='Train Model Imputation')
test_model_bars = ax.bar(x + 2 * width, test_model, width, label='Test Model Imputation')

# Add x-axis labels and tick locations
ax.set_xticks(x+(width/2))
ax.set_xticklabels(models[:5], rotation=45, ha='right')

# Add legend and labels
ax.legend()
ax.set_title('Models test and train performance for Simple imputed and Model imputed data')
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
# Function to add labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels on top of each bar
add_labels(train_simple_bars)
add_labels(test_simple_bars)
add_labels(train_model_bars)
add_labels(test_model_bars)

plt.tight_layout()

# Display the plot
plt.show()



#%%

"""
Removing Outliers further to improve the model performance.
"""


final_df = midf.copy()
sns.boxplot(final_df["price"])

final_df = final_df[(final_df["price"] > 10) & (final_df["price"] < 450)]
final_df2 = final_df[model_cols]

X = final_df2.drop(["price"], axis = 1)
y = final_df2["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
print("After removing the outliers in part2:\n", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Creating a function for modeling experiments.

def model_exp(X_train, y_train, X_test, y_test):

  lr_model = LinearRegression()
  dtr_model_tuned = DecisionTreeRegressor(max_depth = 15, min_samples_split = 40, min_samples_leaf = 10, min_impurity_decrease=0.0001)
  rfr_model_tuned = RandomForestRegressor(n_estimators=100, max_depth = 15, min_samples_split = 40, min_samples_leaf = 10)
  xgb_model = xgb.XGBRegressor()
  lgbm_model = lgbm.LGBMRegressor(num_iterations = 1000, verbose = -1)



  lr_results_midf = train_and_eval(lr_model, X_train, y_train, X_test, y_test)
  dtr_results_tuned_midf = train_and_eval(dtr_model_tuned, X_train, y_train, X_test, y_test)
  rfr_results_tuned_midf = train_and_eval(rfr_model_tuned, X_train, y_train, X_test, y_test)
  xgb_results_midf = train_and_eval(xgb_model, X_train, y_train, X_test, y_test)
  lgbm_results_midf = train_and_eval(lgbm_model, X_train, y_train, X_test, y_test)

  results_midf = pd.DataFrame([lr_results_midf, dtr_results_tuned_midf, rfr_results_tuned_midf, xgb_results_midf, lgbm_results_midf])

  return results_midf

results_3 = model_exp(X_train, y_train, X_test, y_test)


#%%


"""
Compare the model performance
"""

melted_df = results_3[["exp", "Train MAE", "Test MAE"]].melt(id_vars=["exp"], var_name="Type", value_name="MAE")

plt.figure(figsize=(12, 5))
bar_containers = sns.barplot(x="exp", y="MAE", data=melted_df, hue="Type")
plt.xticks(rotation=45)


all_bars = []
for bar_container in bar_containers.containers:
    all_bars.extend(bar_container.get_children())

labels = [f"{bar.get_height():.2f}" for bar in all_bars]


for bar, label in zip(all_bars, labels):
    bar_height = bar.get_height()
    x_coord = bar.get_x() + bar.get_width() / 2
    y_coord = bar_height + bar.get_y() + 0.01  # Adjust the 0.01 value to control label position
    plt.text(x_coord, y_coord, label, ha='center', va='bottom')

plt.title("Model performance on Train and Test sets after outlier removal")
plt.xticks(ticks=melted_df["exp"], rotation=45)

plt.show()


#%%

"""
HYPERPARAMETER TUNING USING OPTUNA
"""

# UNCOMMENT TO RUN THE HYPERPARAMETER TUNING USING OPTUNA
"""
def objective(trial, X_train, y_train, X_test, y_test):
    

    param = {
        'objective': 'mae',
        'random_state': 42,
        'booster': 'gbtree',
        'eta': trial.suggest_float('eta', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_float('gamma', 0, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'tree_method': 'gpu_hist',
        'verbosity': 0
    }

    model = LGBMRegressor(**param, num_iterations=1000, verbose=-1)

    model.fit(X_train, y_train,eval_set=[(X_test,y_test)])

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return mae

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50, n_jobs = -1, show_progress_bar=True)
print('Number of finished trials:', len(study.trials))
print('Best trial:\n', study.best_trial.params)
"""


#%%

# Let's see how the best params improve the performance of the LIGHTGBM model.
#params = study.best_trial.params


# Using the best params from the hyperparameter tuning.
params = {'eta': 0.010491104507019882,
        'subsample': 0.992396132180261,
        'colsample_bytree': 0.9644764422219713,
        'num_parallel_tree': 1,
        'min_child_weight': 32,
        'gamma': 35.0147935853841,
        'max_depth': 10,
        'learning_rate': 0.19732349325787155}

lgbm_hpo = LGBMRegressor(**params, num_iterations = 1000, verbose = -1)
lgbm_results_hpo = train_and_eval(lgbm_hpo, X_train, y_train, X_test, y_test)


# %%


"""
K-FOLD CROSS VALIDATION
"""


scores = []
kf = KFold(n_splits=5, random_state=42, shuffle=True)

X_kf = final_df2.drop("price", axis = 1)
y_kf = final_df2["price"]

for i, (train_index, test_index) in enumerate(kf.split(X_kf)):

    X_tr, y_tr, X_val, y_val = X_kf.iloc[train_index], y_kf.iloc[train_index], X_kf.iloc[test_index], y_kf.iloc[test_index]

    model = LGBMRegressor(**params, num_iterations = 1000, verbose = -1)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)])

    preds = model.predict(X_val)
    score = mean_absolute_error(y_val, preds)
    scores.append(score)

    print(f"Fold{i} Val MAE: {score}")


print(f"5 Fold Average Val MAE: {np.mean(scores)}")


#%%


"""
FINAL MODEL
"""


# City, State are not added early to make the experiments faster and get to a good model. 
# Now adding these columns for a final prediction.


final = final_df[model_cols]

X = final.drop("price", axis = 1)
y = final["price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
print("FINAL shapes: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Fit model
lgbm_model = LGBMRegressor(**params, num_iterations = 1000, verbose = -1)
lgbm_model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluation
lgbm_final_result = display_results(lgbm_model, y_train, train_preds, y_test, test_preds, exp = "FINAL LIGHTGBM MODEL")