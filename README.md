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



**b. Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?**



# **Step 7: Genre and Music Characteristics**

**a. Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?**
**b. Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%**

# **Step 8: Platform Popularity**

**a. How do the numbers of tracks in spotify_playlists, spotify_charts, and apple_playlists compare? Which platform seems to favor the most popular tracks?**

# **Step 9: Advanced Analysis**

**a. Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?**
**b. Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.**

















