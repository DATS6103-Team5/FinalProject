#%%
import pandas as pd
simple_imputed_df = pd.read_csv('simple_imputed_df.csv')


# 1. Identifying the best neighborhoods for a stay in each city

import matplotlib.pyplot as plt
model_imputed_df = model_imputed_df.dropna(subset=['review_scores_rating'])

# Now, calculate the mean review scores rating for each neighborhood
best_neighborhoods = model_imputed_df.groupby(['City', 'neighbourhood_cleansed'])['review_scores_rating'].mean().reset_index()
best_neighborhoods = best_neighborhoods.sort_values(by='review_scores_rating', ascending=False)

# Filter to include only the top 10 neighborhoods for each city
top_10_neighborhoods = best_neighborhoods.groupby('City').head(1)

# Convert 'neighbourhood_cleansed' column to string
top_10_neighborhoods.loc[:, 'neighbourhood_cleansed'] = top_10_neighborhoods['neighbourhood_cleansed'].astype(str)

# Create a bar plot
plt.figure(figsize=(12, 8))
for city, data in top_10_neighborhoods.groupby('City'):
    plt.barh(data['neighbourhood_cleansed'], data['review_scores_rating'], label=city)

# Add labels and title
plt.xlabel('Review Scores Rating')
plt.ylabel('Neighborhood')
plt.title('Top 10 Neighborhoods for a Stay in Each Location')
plt.legend(title='City', loc='upper right')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
plt.tight_layout()
plt.show()


##########################################################################################################################################
# %%
# 2. Determining the top listings in each location

import matplotlib.pyplot as plt
import seaborn as sns

top_listings_sorted = top_listings.sort_values(by='review_scores_rating', ascending=False)

# Convert 'City' column to string
top_listings_sorted['City'] = top_listings_sorted['City'].astype(str)

# Convert 'id' column to string
top_listings_sorted['id'] = top_listings_sorted['id'].astype(str)


# Define a color palette
colors = sns.color_palette("crest", len(top_listings_sorted))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_listings_sorted['City'] + ' (' + top_listings_sorted['id'] + ')', top_listings_sorted['review_scores_rating'], color=colors)
plt.xlabel('Listing (ID)')
plt.ylabel('Review Scores Rating')
plt.title('Top Listings in Each Location')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

###########################################################################################################################################3
#%%
# 3. Recognizing the best host in each location

import pandas as pd

# Group by 'City' and 'host_id', then calculate the mean review scores rating for each host
best_hosts = model_imputed_df.groupby(['City', 'host_id'])['review_scores_rating'].mean().reset_index()

# Sort the DataFrame by review scores rating in descending order
best_hosts = best_hosts.sort_values(by='review_scores_rating', ascending=False)

# Select the best host for each city
best_hosts_in_each_location = best_hosts.groupby('City').head(1)

print("\nBest Hosts in Each Location:")
print(best_hosts_in_each_location)

###########################################################################################################################################
#%%
# 4. Investigating whether neighbourhood significantly impacts price
#Top 10 prices impacted by neighbourhood
import seaborn as sns

# Get the top 10 neighborhoods based on mean price
top_10_neighborhoods = neighborhood_impact.nlargest(10).index

# Filter the data to include only the top 10 neighborhoods
filtered_data = model_imputed_df[model_imputed_df['neighbourhood_cleansed'].isin(top_10_neighborhoods)]

# Define a function to remove outliers based on IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove outliers from the 'price' column
filtered_data = remove_outliers(filtered_data, 'price')

# Create a box plot
plt.figure(figsize=(14, 10))
sns.boxplot(x='neighbourhood_cleansed', y='price', data=filtered_data, palette='viridis')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.xlabel('Neighborhood')
plt.ylabel('Price')
plt.title('Distribution of Prices Across Top 10 Neighborhoods (without outliers)')

# Show plot
plt.tight_layout()
plt.show()


########################################################################################################################################################
#%%
#5. Plotting the locations of listings on a map with the count.

import folium
from folium.plugins import MarkerCluster

# Create a map centered around the mean latitude and longitude
mean_lat = model_imputed_df['latitude'].mean()
mean_lon = model_imputed_df['longitude'].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

# Create a MarkerCluster object
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each listing to the MarkerCluster object
for index, row in model_imputed_df.iterrows():
    folium.Marker([row['latitude'], row['longitude']]).add_to(marker_cluster)
m

#############################################################################################################################################3
#%%
# 6.How have prices changed over time in relation to the number of reviews a listing has received in the last 12 months?

import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for listings with reviews in the last 6 months
recent_reviews = model_imputed_df[model_imputed_df['number_of_reviews_ltm'] > 0]

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='reviews_per_month', y='price', hue='number_of_reviews_ltm', data=recent_reviews, palette='coolwarm')
plt.title('Price vs. Reviews per Month')
plt.xlabel('Reviews per Month')
plt.ylabel('Price')
plt.legend(title='Number of Reviews (Last 12 Months)')
plt.show()
