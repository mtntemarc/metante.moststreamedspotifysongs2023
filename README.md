# Exploratory Data Analysis on the Most Streamed Spotify 2023 Dataset

# I. INTRODUCTION
This task involves an exploratory data analysis (EDA) of Spotify’s most-streamed songs in 2023 to uncover patterns and insights into track popularity. The process starts by reviewing the dataset for missing values and data types, followed by summary statistics highlighting metrics like stream counts and musical attributes (e.g., BPM, danceability). Visualizations, such as bar charts and scatter plots, will identify trends, while correlations between streams and features like tempo and energy will provide insights into factors driving popularity. Findings will be summarized with recommendations to clarify what makes a track successful on Spotify in 2023.

In this project, I took a step-by-step approach to explore and analyze the dataset, ensuring each part of the process was thorough and clear. I started by outlining key questions to focus on the most relevant insights. From there, I followed a detailed coding process to clean, visualize, and examine the data carefully. By the end, I could draw out some interesting trends and patterns, bringing out insights from the results. This method helped me break down the analysis in a structured way.

# II. Coding Process 

# **Step 1: Importing Libraries**
- We start by importing libraries that give us the tools we need for organizing, analyzing, and visualizing the data, making the whole analysis process smoother and more efficient.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
# **Step 2: Loading the CSV File**

What is the purpose of using ISO-8859-1?
- When you use ```pd.read_csv ``` without setting the encoding, it assumes the file is in UTF-8, which can handle characters from many languages around the world. If your file has special European characters (like é or ñ) and UTF-8 causes errors, using ```ISO-8859-1``` can help because it’s designed specifically for Western European characters, making it easier to read files with those accents correctly.

```
df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')
df
```
![image](https://github.com/user-attachments/assets/0dbb4286-daa1-4649-863e-63e037ac50a9)

# **Step 3: Overview of the Dataset**

**a. How many rows and columns does the dataset contain?**
- ```df.shape ```is an attribute of a pandas DataFrame that shows the dimensions of the dataset.

```
df.shape
```

![image](https://github.com/user-attachments/assets/21b89512-eebb-4be4-a134-be959b595e8b)

**b. What are the data types of each column? Are there any missing values?**
- In determining the data types of each column ``` dtypes ``` attribute is used.
```
df.dtypes

```

![image](https://github.com/user-attachments/assets/27a69b15-5bea-4d1f-8107-f60dfdef7596)

**Are there any missing values?**
- To find any missing values use ``` .isnull().sum() ```.

```
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/ec6146e4-5836-417e-9d1f-c727186f8762)

Columns with non-zero values indicate missing entries. For example, ``` in_shazam_charts ``` has 50 missing values, and ``` key ```has 95, meaning data is absent in those rows. Identifying these gaps helps determine where data cleaning is needed to ensure a complete analysis.

# **Step 4: Basic Descriptive Statistics**

**a. What are the mean, median, and standard deviation of the streams column?**

- Convert First Non-Numeric Values in streams. This is important because if there are values that can't be converted (like text or special characters), they’ll be set to NaN (missing values) due to errors='coerce'. This avoids calculation errors in the next steps.

```
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
```

**Calculate Mean:**

```
s_mean = df['streams'].mean()
```

**Calculate Median:**

```
s_median = df['streams'].median()
```

**Calculate Standard Deviation:**

```
s_std = df['streams'].std()
```
**Print Results:**

```
print('Mean:', s_mean)
print('Median:', s_median)
print('Standard Deviation:', s_std)
```

![image](https://github.com/user-attachments/assets/2978f06a-8e04-4e74-8e22-8e8fc015000c)

- The streams column has a mean of 514,137,424.94 and a median of 290,530,915.0, indicating that a few highly streamed songs raise the average. The standard deviation is 566,856,949.04, showing significant variability, with some songs having far more streams than others, suggesting a few very popular songs among many with lower streams.


**b. What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?**

- To analyze the distribution of released_year and artist_count, we can use histograms, along with summary statistics.

**Released Year Histogram**

```
#Create a histogram for 'released_year'

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='released_year', color='blue', kde=True)
plt.title('Distribution of Release Years', fontsize=16, weight='bold')
plt.xlabel('Year of Release', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
sns.despine() #Remove unnecessary spines

#Output

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/16afd21f-3ccf-4100-bc6b-2212b430f573)

This shows how many songs were released each year. If certain years had more popular songs than others, we might look for trends such as an increase in popular song releases in recent years or a concentration of popular tracks from a specific period.

**Artist Count**

```
#Create a histogram for 'artist_count' with KDE for smoother distribution
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='artist_count', color='red', kde=True)
plt.title('Distribution of Artist Counts', fontsize=16, weight='bold')
plt.xlabel('Artist Count', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine() #Remove unnecessary spines for a cleaner appearance

#Output
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/beb28680-ec4f-426c-939c-ede28bfe70ab)

This represents the number of artists involved in each song, the graph helps illustrate if most songs are collaborations or solo performances. For example, if ```artist_count``` shows a high number of singles or a trend toward collaborations with multiple artists, it could suggest a trend toward collaboration in popular music.

**Are there any noticeable trends or outliers?**

**Scatter Plot for Released Year Histogram and Artist Count** 

![image](https://github.com/user-attachments/assets/6b5a575d-815e-47d4-8af6-8b7c671eb7ef)

The plot reveals an evolving trend in popular music, with older tracks generally involving fewer artists and newer tracks showing more collaborative efforts. This trend highlights the growing popularity of multi-artist collaborations in the music industry.

# **Step 5: Top Performers**

**a. Which track has the highest number of streams? Display the top 5 most streamed tracks.**

- Sort the streams column in descending order to identify the highest-streamed songs and then select the top 5 entries, display using ```head()```.

```
#Sort streams in descending order
top_streams = df.sort_values(by='streams', ascending=False).head()

#Show Top 5 using head 
top_streams.head()

```

![image](https://github.com/user-attachments/assets/e60110c6-71f7-4713-b439-34b18401cf3c)



**b. Who are the top 5 most frequent artists based on the number of tracks in the dataset?**

- Identify each artist's appearances in the dataset, select the top 5 most frequent, and print each artist with their appearance count.

```
# Count the frequency of each artist and get the top 5

artist_freq = df['artist(s)_name'].value_counts().head()

#Display top 5 most frequent artists and their frequencies
print("Top 5 Most Frequent Artists:\n")

for artist, count in artist_freq.items(): #Iterates through each artist and their count in artist_freq

    #Output
    print(f"{artist}: {count} appearances")

``` 
![image](https://github.com/user-attachments/assets/ccac6f5c-7f1c-49f4-ba4f-c267d067cfd6)

# **Step 6: Temporal Trends**

**a. Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.**
```
#The number of tracks released per year and sort by year
tracks_yearly = df['released_year'].value_counts().sort_index()

#Plot using Line Plot 
plt.figure(figsize=(12, 6))
sns.lineplot(data=tracks_yearly, color='red', marker='o', markersize=8, linewidth=2)
plt.title('Number of Tracks Released Per Year', fontsize=18, weight='bold')
plt.xlabel('Release Year', fontsize=15)
plt.ylabel('Number of Tracks', fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

#Output
plt.tight_layout()
plt.show()

```
![image](https://github.com/user-attachments/assets/3f9b5b8e-e877-4f97-be08-b09a9a38bab8)

- The plot shows a sharp increase in the number of tracks released per year, particularly after 2020, indicating a recent surge in music releases.


**b. Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?**
```
#Count the number of tracks released per month and sort by month index
monthly_tracks = df['released_month'].value_counts().sort_index()
print(monthly_tracks)

#Names of the months in order
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

#Number of tracks released per month
plt.figure(figsize=(14, 7))
sns.barplot(x=month_names, y=monthly_tracks.values,)
plt.title('Tracks Released Per Month', fontsize=18, weight='bold', color='darkblue')
plt.xlabel('Month', fontsize=15, labelpad=10)
plt.ylabel('Number of Tracks', fontsize=15, labelpad=10)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

#Output
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/49b8e0b9-7716-495d-b7a2-8eb631a94d6a)

- The chart shows that January and May have the highest number of track releases, likely to influence new-year new music interest and the lead-up to summer. Releases go down mid-year, especially in July and August, and increase again in October and November, possibly targeting the holiday season. This suggests a pattern where music releases align with seasonal trends and consumer demand.


# **Step 7: Genre and Music Characteristics**

**a. Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?**

**Correlation Matrix between All Variables in Dataset**

```
#Select following datatypes to include in matrix
numeric_df = df.select_dtypes(include=['float64' ,'int64'])

#Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

#Plot the correlation matrix using Seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between All Variables in Dataset', fontsize=18, weight='bold', color='midnightblue')

#Output
plt.tight_layout()
plt.show()


```
![image](https://github.com/user-attachments/assets/66d93894-985e-48d0-ac1f-b6df93af139a)
- The correlation matrix reveals that playlist and chart placements strongly influence a song's streams. Specifically, being in Spotify Playlists, Apple Playlists, and Spotify Charts shows the highest positive correlation with stream counts. On the other hand, musical attributes like BPM, Danceability%, and Energy% have very weak or negligible correlations with streams, suggesting that these characteristics don’t significantly impact a song's popularity. In short, exposure through playlists and charts is far more crucial for streaming success than the song's musical features.


**b. Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%**

**Correlation between danceability_% and energy_%**
```
#Correlation between danceability_% and energy_%

# Plotting the scatter plot for danceability_% vs energy_%
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cleaned, x='danceability_%', y='energy_%', color='RED', alpha=0.6, edgecolor=None)
plt.title('Scatter Plot of Danceability% vs Energy%', fontsize=15, weight='bold')
plt.xlabel('Danceability%', fontsize=12)
plt.ylabel('Energy%', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


#Output
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/f2ac81a2-fb79-4f81-b963-5e0490705609)
- The scatter plot reveals that songs in this dataset generally fall into the mid-range for both danceability and energy, with most clustered between 50-70% for danceability and 50-80% for energy. There isn’t a clear relationship between the two; songs with high energy don’t necessarily have high danceability, and vice versa. It seems that most tracks are fairly balanced, with moderate levels of both qualities, but there's plenty of variety across the board.

**Correlation between valence_% and acousticness_%**

```
# Correlation between valence_% and acousticness_%
# Plotting the scatter plot for valence_% vs acousticness_%
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cleaned, x='valence_%', y='acousticness_%', color='darkblue', alpha=0.6, edgecolor=None)
plt.title('Scatter Plot of Valence% vs Acousticness%', fontsize=15, weight='bold')
plt.xlabel('Valence%', fontsize=12)
plt.ylabel('Acousticness%', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

#Output
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/3ba55897-1593-4329-8df2-cfc99a7e4bd0)
- The plot shows that as Valence% goes up, Acousticness% tends to go down. In other words, songs that feel happier or more positive are often less acoustic, possibly favoring electronic or synthesized sounds. This aligns with the moderate negative correlation we observed, highlighting a subtle pattern in music production choices.


# **Step 8: Platform Popularity**

**a. How do the numbers of tracks in spotify_playlists, spotify_charts, and apple_playlists compare? Which platform seems to favor the most popular tracks?**

**Distribution of Tracks Across Combined Platforms and Charts**
```
#Calculate the total track counts by combining playlists and charts for each platform


#The .sum() method adds up all the values in each specified column
total_combined_counts = {
    'Spotify (Playlists + Charts)': df['in_spotify_playlists'].sum() + df['in_spotify_charts'].sum(),
    'Apple (Playlists + Charts)': df['in_apple_playlists'].sum() + df['in_apple_charts'].sum(),
    'Deezer (Playlists + Charts)': df['in_deezer_playlists'].sum() + df['in_deezer_charts'].sum(),
    'Shazam Charts': df['in_shazam_charts'].sum()
}

#Create a pie chart to display the combined data for each platform

plt.figure(figsize=(10, 8))
plt.pie(
    total_combined_counts.values(),        
    labels=total_combined_counts.keys(),      # Labels for each section
    autopct='%1.1f%%',                        # Display percentages on each section with 1 decimal plac
    startangle=140,                          
    colors=['skyblue', 'red', 'yellow', 'green'] )
plt.title('Distribution of Tracks Across Combined Platforms and Charts', fontsize=15, weight='bold')

#Output
plt.show()

```

![image](https://github.com/user-attachments/assets/c27a05b1-aebf-4416-b9e9-f98786089be3)
- The pie chart shows that Spotify is the dominant platform for music distribution, with an impressive 95.1% of all tracks appearing in its playlists and charts. In comparison, Apple’s playlists and charts hold a much smaller share at 3.2%, followed by Deezer with 1.5%, and Shazam with just 0.1%. This distribution highlights how influential Spotify is in the streaming world, with most tracks reaching listeners through its platform.


**Which platform seems to favor the most popular tracks?**
```
#Show top 10 tracks by sorting the DataFrame by 'streams' in descending order
top_ten_tracks = df.sort_values(by='streams', ascending=False).head(10)

#Select specific columns for display
top_ten_tracks_out = top_ten_tracks.loc[:, [
    'track_name',            
    'streams',                 
    'in_spotify_playlists',    
    'in_spotify_charts',       
    'in_apple_playlists',      
    'in_apple_charts',        
    'in_deezer_playlists',    
    'in_deezer_charts',      
    'in_shazam_charts'         
]]

#Output
top_ten_tracks_out
```
![image](https://github.com/user-attachments/assets/4407466d-87ec-4f41-8fa7-86352cd99924)
- Spotify favors the most popular tracks, as seen by their high representation in both playlists and charts on this platform, significantly more than on Apple, Deezer, or Shazam.

# **Step 9: Advanced Analysis**

**a. Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?**
```
#Summing streams for each key by grouping data into minor and major modes
minor = df[df['mode'] == 'Minor'].groupby(['key'])['streams'].sum()
major = df[df['mode'] == 'Major'].groupby(['key'])['streams'].sum()

# Setting the data for the plot
x = np.arange(len(minor))  
keys = minor.index.tolist()  
minorbar = minor.values  # Total streams for each key in minor mode
majorbar = major.values  # Total streams for each key in major mode

#Plotting the Graph
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, majorbar, width=0.4, label="Major", linewidth=0.7, color='blue')
plt.bar(x + 0.2, minorbar, width=0.4, label="Minor", linewidth=0.7, color='skyblue') 
plt.title("Streams by Key and Mode")
plt.xticks(x, keys)  # Set x-axis labels to the unique keys
plt.xlabel("Key")
plt.ylabel("Streams")
plt.legend(title="Mode")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

#Output
plt.show()

```
![image](https://github.com/user-attachments/assets/ef3aab91-0cc1-4109-ac56-6ee1f3ab8396)

- The data shows that tracks in certain keys, especially C# and D in Major mode, tend to have higher streams, hinting at a listener preference for these musical qualities. Overall, tracks in Major mode generally outperform those in Minor mode, suggesting that the brighter sound of Major keys might be more popular with audiences.



**b. Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.**

```
#Convert playlist and chart columns to numeric, setting errors='coerce' to handle non-numeric values
playlist_columns = ['in_spotify_playlists', 'in_deezer_playlists', 'in_apple_playlists']
chart_columns = ['in_spotify_charts', 'in_deezer_charts', 'in_apple_charts', 'in_shazam_charts']

df[playlist_columns + chart_columns] = df[playlist_columns + chart_columns].apply(pd.to_numeric, errors='coerce')

#Replace NaN values with 0 in the playlist and chart columns to allow summing
df[playlist_columns + chart_columns] = df[playlist_columns + chart_columns].fillna(0)

# Get top 10 most frequently appearing artists and convert to a list
top_artists = df['artist(s)_name'].value_counts().head(10).index.tolist()

#Filter the DataFrame to only include rows for the top 10 artists
top_df = df[df['artist(s)_name'].isin(top_artists)]

#Summing playlist counts across platforms for each artist
top_playlists = top_df.groupby('artist(s)_name')[playlist_columns].sum()
top_playlists['Total playlists'] = top_playlists.sum(axis=1)  
top_playlists = top_playlists.sort_values(by='Total playlists', ascending=False)  #Sort by total playlists, descending
top_charts = top_df.groupby('artist(s)_name')[chart_columns].sum()
top_charts['Total charts'] = top_charts.sum(axis=1)  
top_charts = top_charts.reindex(top_playlists.index)  #Align with the playlist DataFrame order

#Prepare data for plotting
x_labels = np.arange(10)  
artist_names = top_playlists.index.tolist()  #Extract artist names for x-axis labels
playlist_counts = top_playlists['Total playlists'].values  #Get total playlists values
chart_counts = top_charts['Total charts'].values  #Get total charts values

plt.figure(figsize=(10, 6))
plt.bar(x_labels - 0.2, playlist_counts, width=0.4, label="Playlists", linewidth=0.7, color='pink')
plt.bar(x_labels + 0.2, chart_counts, width=0.4, label="Charts", linewidth=0.7, color='red')
plt.title("Playlists and Charts of the Top 10 Frequently Appearing Artists")
plt.xticks(x_labels, artist_names, rotation=45) 
plt.xlabel("Artists")
plt.ylabel("Count")
plt.legend() 
plt.tight_layout()  

#Output
plt.show()
```
![image](https://github.com/user-attachments/assets/b6dd2c3b-7cec-4ede-9d73-c4a9a54bf9ae)

- The chart reveals that artists like The Weeknd, Taylor Swift, Ed Sheeran, and Harry Styles consistently appear in playlists far more than in charts, indicating strong playlist popularity but lower chart dominance. These artists likely have broad appeal and significant fan bases, leading to frequent playlist additions across platforms. However, their relatively low chart appearances suggest that while their music is widely accessible and continuously streamed, it doesn’t necessarily remain at the top of charts as consistently. This highlights a difference between playlist and chart dynamics for popular artists.


# III. Conclusion 

In this analysis of Spotify's top-streamed songs in 2023, we explored trends and patterns that drive music popularity. We started by organizing and cleaning the dataset, then examined key metrics, like how track releases have increased in recent years, especially during certain months that align with seasonal trends.

When we looked at what influences streaming numbers, we found that playlist and chart placements on platforms like Spotify are far more important than musical features like BPM or energy level. This highlights the power of exposure on major platforms over specific song characteristics.

Additionally, we found that some artists, including The Weeknd, Taylor Swift, Ed Sheeran, and Harry Styles, consistently appear more often in playlists than on charts. This shows they have widespread appeal and are frequently added to playlists, even if they don’t always top the charts. Overall, this analysis underscores the importance of playlisting in boosting streams, with Spotify standing out as a key player in music distribution.


# IV. References

Malli. (2024, March 28). Use pandas.to_numeric() Function. Spark by Examples. https://sparkbyexamples.com/pandas/use-pandas-to-numeric-function/#:~:text=Key%20Points%20%E2%80%93-,Pandas.,errors%2C%20coercion%2C%20and%20downcasting.

Ahmed, A. R. A. (2023). Most Streamed Spotify Songs [Data set]. Kaggle. https://www.kaggle.com/code/ahmedredaahmedali/most-streamed-spotify-songs

DataDaft. (2022, March 11). Code with Data - Histogram in Matplotlib [Video]. YouTube. https://youtu.be/0Ddzm6PpkI0


# Marc Gabriel G. Metante 2ECE-B 

# 


















