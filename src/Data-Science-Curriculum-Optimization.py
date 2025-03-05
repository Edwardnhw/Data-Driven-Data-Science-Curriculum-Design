# %% [markdown]
# # Creating University Program Curriculumn based on Clustering of Skills in Job Postings 
# 

# %% [markdown]
# ### Import Python libraries


# %%

import numpy as np
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt



# %%
import nltk
nltk.download('punkt')


# %%
## Machine Learning libraries
import numpy as np
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import ClusterWarning

zeros = np.zeros 

# %% [markdown]
# ### Load job postings web-scraped from Indeed.com

# %%
# Sample dataset file name
filename_data = 'data/webscraping_results.csv'

## Read csv file (dataset)
results = pd.read_csv(filename_data, low_memory=False)

#Change column names
results.rename(columns={'Job_Title': 'Title', 'Company_Name': 'Company', 'Job_Description': 'Descriptions'}, inplace=True)

# %% [markdown]
# ### Extract skills from job postings
# 
# Replace with your own classification of skills. You may use ChatGPT to generate Python code for extracting skills from job descriptions.

# %%
# Initialize skill dictionaries with empty lists for binary flags
sskills = {key: [] for key in ['Python', 'Matlab', 'Kotlin', 'Java', 'C++', 'SQL', 'Microsoft Excel', 'TensorFlow', 'PyTorch']}
tskills = {key: [] for key in ['Data Management', 'Big Data', 'Data Analysis', 'Machine Learning', 'NLP', 
                               'Data Visualization', 'Cloud Computing', 'Statistics', 'Modeling']}
bskills = {key: [] for key in ['Project Management', 'Consulting', 'Negotiation', 'Strategic Thinking',
                               'Business Intelligence', 'Market Analysis', 'Risk Management']}
pskills = {key: [] for key in ['Teamwork', 'Creativity', 'Communication', 'Leadership', 'Time Management', 'Adaptability']}

# Extract skills from job postings
for ir, dfr in results.iterrows():
    cleantext = str(dfr["Descriptions"]).lower()

    # Ensure every skill key gets a value ('1' or '0') in this iteration
    # Programming/System Skills
    sskills['Python'].append('1' if 'python' in cleantext else '0')
    sskills['Matlab'].append('1' if 'matlab' in cleantext else '0')
    sskills['Kotlin'].append('1' if ' Kotlin ' in cleantext else '0')
    sskills['Java'].append('1' if 'java' in cleantext else '0')
    sskills['C++'].append('1' if 'c++' in cleantext else '0')
    sskills['SQL'].append('1' if 'sql' in cleantext else '0')
    sskills['Microsoft Excel'].append('1' if 'microsoft excel' in cleantext else '0')
    sskills['TensorFlow'].append('1' if 'tensorflow' in cleantext else '0')
    sskills['PyTorch'].append('1' if 'pytorch' in cleantext else '0')

    # Technical/Data-Related Skills
    tskills['Data Management'].append('1' if 'data management' in cleantext else '0')
    tskills['Big Data'].append('1' if 'big data' in cleantext else '0')
    tskills['Data Analysis'].append('1' if 'data analysis' in cleantext else '0')
    tskills['Machine Learning'].append('1' if 'machine learning' in cleantext else '0')
    tskills['NLP'].append('1' if 'nlp' in cleantext else '0')
    tskills['Data Visualization'].append('1' if 'data visualization' in cleantext else '0')
    tskills['Cloud Computing'].append('1' if 'cloud computing' in cleantext else '0')
    tskills['Statistics'].append('1' if 'statistics' in cleantext else '0')
    tskills['Modeling'].append('1' if 'modeling' in cleantext else '0')

    # Business Intelligence/Project Management Skills
    bskills['Project Management'].append('1' if 'project management' in cleantext else '0')
    bskills['Consulting'].append('1' if 'consulting' in cleantext else '0')
    bskills['Negotiation'].append('1' if 'negotiation' in cleantext else '0')
    bskills['Strategic Thinking'].append('1' if 'strategic thinking' in cleantext else '0')
    bskills['Business Intelligence'].append('1' if 'business intelligence' in cleantext else '0')
    bskills['Market Analysis'].append('1' if 'market analysis' in cleantext else '0')
    bskills['Risk Management'].append('1' if 'risk management' in cleantext else '0')

    # Teamwork and Communication Skills
    pskills['Teamwork'].append('1' if 'teamwork' in cleantext else '0')
    pskills['Creativity'].append('1' if 'creativity' in cleantext else '0')
    pskills['Communication'].append('1' if 'communication' in cleantext else '0')
    pskills['Leadership'].append('1' if 'leadership' in cleantext else '0')
    pskills['Time Management'].append('1' if 'time management' in cleantext else '0')
    pskills['Adaptability'].append('1' if 'adaptability' in cleantext else '0')

# Sanity Check: Ensure all lists are of the same length
for skill_dict in [sskills, tskills, bskills, pskills]:
    for key, value in skill_dict.items():
        if len(value) != len(results):
            raise ValueError(f"Key '{key}' in dictionary has inconsistent length: {len(value)} vs {len(results)}")

# Combine all extracted skills into DataFrames
df1 = results[['Title', 'Company', 'Location', 'Descriptions']].copy()
df2 = pd.DataFrame(sskills)
df3 = pd.DataFrame(tskills)
df4 = pd.DataFrame(bskills)
df5 = pd.DataFrame(pskills)

# Concatenate all DataFrames horizontally
final_df = pd.concat([df1, df2, df3, df4, df5], axis=1)

# Display the resulting DataFrame
print(final_df.head())


# %%
## Create dataframe with extracted skills (1 if a skill was found in job description, 0 if a skills was not found in job description)
df1 = results[['Title', 'Company', 'Location', 'Descriptions']].copy()
df2 = pd.DataFrame(sskills)
df3 = pd.DataFrame(tskills)
df4 = pd.DataFrame(bskills)
df5 = pd.DataFrame(pskills)
frames = [df1, df2, df3, df4, df5]
res = pd.concat(frames, axis = 1)
res.head()

# %%
## Save skills as 2D array
df = res.iloc[:,4:]
df_summary = df.apply(pd.to_numeric)
a = df_summary.values

print("Number of job postings:", a.shape[0])
print(a)

# %% [markdown]
# ### Feature Engineering for visualization

# %%
import pandas as pd
import re

def preprocess_csv(filepath):
    # Load the dataset
    results = pd.read_csv(filepath)
    
    # Handle NaN values globally
    results['Descriptions'] = results['Descriptions'].fillna('')
    results['Title'] = results['Title'].fillna('Unknown Title')
    results['Company'] = results['Company'].fillna('Unknown Company')
    results['Location'] = results['Location'].fillna('Unknown Location')
    results['Salary'] = results['Salary'].fillna('')
    
    # Standardize Salary
    def parse_salary(salary):
        if pd.isnull(salary) or salary == '':
            return None
        match = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', salary)
        if not match:
            return None
        salary_range = [float(s.replace(',', '')) for s in match]
        avg_salary = sum(salary_range) / len(salary_range)
        if 'an hour' in salary:
            return avg_salary * 40 * 52
        elif 'a day' in salary:
            return avg_salary * 5 * 52
        elif 'a week' in salary:
            return avg_salary * 52
        elif 'a month' in salary:
            return avg_salary * 12
        else:
            return avg_salary

    results['Standardized_Salary'] = results['Salary'].apply(parse_salary)
    results['Standardized_Salary'] = results['Standardized_Salary'].fillna(results['Standardized_Salary'].median())
    
    return results

#  Usage
results = preprocess_csv('data/webscraping_results.csv')


# %%
import re
from nltk.tokenize import sent_tokenize

def extract_education_level(description):
    """
    Extract the highest education level mentioned in the description,
    ensuring exact matches for phrases like 'graduate degree' and 'undergraduate degree'.
    """
    # Define the hierarchy and keywords with exact matching
    education_keywords = {
        'Doctoral Degree': [r'\bphd\b', r'\bdoctorate\b', r'\bdoctoral\b', r'\bpostdoctoral\b'],
        'Master’s Degree': [
            r"\bmaster's degree\b", r'\bgraduate degree\b', r'\bmba\b', r'\bmsc\b',
            r'\bmaster of science\b', r'\badvanced degree\b'
        ],
        'Bachelor’s Degree': [
            r"\bbachelor's degree\b", r'\bundergraduate degree\b', r'\bba/bs\b',
            r'\bbachelor of arts\b', r'\bbachelor of science\b'
        ],
        'Associate’s Degree': [r"\bassociate's degree\b", r'\b2-year degree\b'],
        'High School Diploma': [r'\bhigh school diploma\b', r'\bged\b', r'\bno degree required\b']
    }

    # Search for exact matches in hierarchical order
    for level, patterns in education_keywords.items():
        for pattern in patterns:
            if re.search(pattern, description):
                return level

    return 'Unspecified'

def extract_relevant_sentences(description):
    """
    Extract sentences containing education-related terms for focused analysis.
    """
    sentences = sent_tokenize(description)
    relevant_sentences = [sentence for sentence in sentences if any(
        keyword in sentence.lower() for keyword in ['qualification', 'education', 'degree']
    )]
    return " ".join(relevant_sentences)

def process_with_exact_matching(description):
    """
    Process description with sentence extraction and exact phrase matching.
    """
    # Extract relevant sentences
    relevant_context = extract_relevant_sentences(description)
    # Match the highest education level using refined logic
    return extract_education_level(relevant_context)

# Apply the process_with_exact_matching function
results['Education_Level'] = results['Descriptions'].apply(process_with_exact_matching)

# Ensure all categories are represented in the frequency table
education_levels = ['Doctoral Degree', 'Master’s Degree', 'Bachelor’s Degree', 'High School Diploma']
frequency_table = results['Education_Level'].value_counts().reindex(education_levels, fill_value=0)

# Visualization
def visualize_education_levels(frequency_table):
    """
    Visualize the education level distribution, ensuring all categories are present.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(frequency_table.index, frequency_table.values, color='skyblue')
    plt.title('Distribution of Education Levels')
    plt.xlabel('Education Level')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

visualize_education_levels(frequency_table)



# %%
# Extended list of keywords for remote jobs
remote_keywords = (
    r'remote|work from home|telecommute|flexible location|distributed team|location-independent|'
    r'anywhere|virtual role|home-based|online role|work from anywhere|remote-friendly|non-location specific'
)

# Feature: Percentage of Remote Jobs
results['Is_Remote'] = results['Descriptions'].str.contains(
    remote_keywords, case=False, na=False
) | results['Location'].str.contains(
    remote_keywords, case=False, na=False
)

# Count remote and non-remote jobs
remote_counts = results['Is_Remote'].value_counts()

# Visualization
plt.figure(figsize=(6, 6))
plt.pie(remote_counts, labels=['Non-Remote', 'Remote'], autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title('Percentage of Remote Jobs (Based on Descriptions and Location)')
plt.show()


# %%

# Refine FAANG filter
faang_companies = ['facebook', 'meta', 'amazon', 'apple', 'netflix', 'google']
results['Is_FAANG'] = results['Company'].str.contains('|'.join(faang_companies), case=False, na=False)

# Skills to track 
skills = list(sskills.keys()) + list(tskills.keys()) + list(bskills.keys()) + list(pskills.keys())

# Skill counts with filtering for overmatching
faang_skill_counts = {}
for skill in skills:
    faang_skill_counts[skill] = results[results['Is_FAANG']]['Descriptions'].str.contains(
        rf'\b{skill}\b', case=False, na=False
    ).sum()

# Convert to DataFrame for visualization
faang_skill_counts_df = pd.DataFrame.from_dict(faang_skill_counts, orient='index', columns=['Frequency'])

# Visualization
plt.figure(figsize=(10, 6))
faang_skill_counts_df.sort_values(by='Frequency', ascending=False).head(10).plot(
    kind='bar', color='orange', legend=False
)
plt.title('Top Skills Mentioned in FAANG Postings')
plt.xlabel('Skills')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Step 1: Define hard skills to track (from extracted features)
hard_skills = list(sskills.keys())  # Focus on programming/system-related hard skills

# Step 2: Count occurrences of each hard skill in job descriptions
skill_counts = {
    skill: results['Descriptions'].str.contains(skill, case=False, na=False).sum() for skill in hard_skills
}

# Step 3: Convert to DataFrame for visualization
skill_counts_df = pd.DataFrame.from_dict(skill_counts, orient='index', columns=['Frequency'])

# Step 4: Visualization
plt.figure(figsize=(10, 6))
skill_counts_df.sort_values(by='Frequency', ascending=False).plot(kind='bar', color='lightblue', legend=False)
plt.title('Frequency of Hard Skills in Job Descriptions')
plt.xlabel('Skills')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Hierarchical clustering of skills

# %% [markdown]
# Import Machine Learning libraries in Python

# %%
## Create empty matrix to fill
D = np.zeros([a.shape[1],a.shape[1]])

## Find all element-wise skill proximities (distances)
for k in range(a.shape[0]):
    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
             D[i, j] += a[k, i] * a[k, j]
             
# This step ensures the matrix is within a consistent range
max_value = np.max(D)
if max_value > 0:  # Avoid division by zero
    D = D / max_value

print("Proximity Matrix:")
print(D)

# %%
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Validate the distance matrix
if np.any(np.isnan(D)) or np.any(np.isinf(D)):
    raise ValueError("Distance matrix contains NaN or Inf values.")

# Validate labels
assert len(df_summary.columns) == D.shape[0], "Number of labels must match number of rows in the distance matrix."

# Create linkage matrix
Y = sch.linkage(D, method='complete')  # Use 'complete', 'average', or 'ward' as needed


# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 12))
Z = sch.dendrogram(Y, orientation='right', ax=ax, labels=df_summary.columns)
plt.savefig('dendrogram.png', format='png', bbox_inches='tight')
plt.show()



# %% [markdown]
# ### Print clusters for selected distance level

# %%
from scipy.cluster.hierarchy import fcluster

# Set the max_d threshold for cutting the dendrogram
max_d = 0.3  # Adjust this value based on your dataset and dendrogram visualization

# Generate clusters
clusters = fcluster(Y, t=max_d, criterion='distance')

print(f"Clusters at max_d={max_d}: {clusters}")

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 12))
Z = sch.dendrogram(Y, orientation='right', ax=ax, labels=df_summary.columns)
labels = df_summary.columns[Z['leaves']]

# Add a vertical line for max_d
plt.axvline(x=max_d, c='k', linestyle='--', label=f"Cut at max_d={max_d}")

# Save and show the plot
plt.legend(loc='upper right')
plt.savefig('dendrogram.png', format='png', bbox_inches='tight')
plt.show()

# %%
## Identify clusters with max_d cut

lbs = sch.fcluster(Y, max_d*D.max(), 'distance')
clustr = lbs[Z['leaves']]

clust_skls = {}
for k in list(set(clustr)):
    clust_skls[k] = []

for j in range(len(labels)):
    clust_skls[clustr[j]].append(labels[j])

# %%
for key, value in clust_skls.items():
    print(key, value)

# %%
print("Number of automatically created clusters:",len(clust_skls))

# %% [markdown]
# ### Manually adjust clusters before analysis (if necessary)

# %%
clust_skills = {}
clust_skills[0] = ['Strategic Thinking', 'Negotiation', 'Market Analysis', 'Project Management', 'Risk Management', 'Leadership']
clust_skills[1] = ['Cloud Computing', 'Big Data', 'Hadoop', 'Spark', 'SQL', 'Data Management']
clust_skills[2] = ['Matlab', 'Python', 'Kotlin', 'Java', 'C++', 'Programming for Applied AI']
clust_skills[3] = ['Time Management', 'Adaptability', 'Creativity', 'Communication', 'Teamwork', 'Personal Productivity']
clust_skills[4] = ['Data Visualization', 'Tableau', 'Power BI', 'Data Storytelling']
clust_skills[5] = ['TensorFlow', 'PyTorch', 'Machine Learning', 'Deep Learning', 'Artificial Intelligence', 'Neural Networks']
clust_skills[6] = ['Statistics', 'Modeling', 'Data Analysis', 'NLP', 'Optimization', 'Advanced Analytics']
clust_skills[7] = ['Consulting', 'Business Intelligence', 'Strategy Implementation']


# %%
len(clust_skills)
print("Number of manually adjusted clusters:",len(clust_skills))

# %%
section_3_curriculum_sequence = {
    "Semester 1": [
        {"Course": "Foundations of Data Management and Business Intelligence", "Topics": ["Data Storage and Retrieval", "SQL", "Business Intelligence Tools"]},
        {"Course": "Programming for Data Science", "Topics": ["Python", "Advanced SQL", "Version Control (Git)"]},
    ],
    "Semester 2": [
        {"Course": "Statistical Foundations and Data Analysis", "Topics": ["Descriptive and Inferential Statistics", "Hypothesis Testing", "Exploratory Data Analysis"]},
        {"Course": "Data Visualization and Storytelling", "Topics": ["Tableau", "Power BI", "Data Storytelling Techniques"]},
    ],
    "Semester 3": [
        {"Course": "Machine Learning and Predictive Modeling", "Topics": ["Regression and Classification", "Clustering", "Model Evaluation and Tuning"]},
        {"Course": "Business Strategy and Market Analysis", "Topics": ["Strategic Thinking", "Market Analysis", "Risk Management"]},
    ],
    "Semester 4": [
        {"Course": "Advanced Machine Learning and AI", "Topics": ["TensorFlow", "PyTorch", "Natural Language Processing"]},
        {"Course": "Cloud Computing and Big Data Engineering", "Topics": ["AWS/Azure/GCP", "Hadoop", "Big Data Processing"]},
    ],
    "Semester 5": [
        {"Course": "Leadership and Team Management", "Topics": ["Leadership Skills", "Team Collaboration", "Project Management"]},
        {"Course": "Ethics and Governance in AI", "Topics": ["Ethical AI", "Data Privacy", "Regulatory Compliance"]},
    ],
    "Semester 6": [
        {"Course": "Business Intelligence and Consulting", "Topics": ["Consulting Techniques", "Strategy Implementation", "Negotiation"]},
        {"Course": "Capstone Project", "Topics": ["Integrated Application of All Skills"]},
    ],
}


# %% [markdown]
# ### K-means clustering implementation

# %% [markdown]
# ### Feature Engineering for Clustering

# %%
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

# Define skills groups
sskills = {key: [] for key in ['Python', 'Matlab', 'Kotlin', 'Java', 'C++', 'SQL', 'Microsoft Excel', 'TensorFlow', 'PyTorch']}
tskills = {key: [] for key in ['Data Management', 'Big Data', 'Data Analysis', 'Machine Learning', 'NLP', 
                               'Data Visualization', 'Cloud Computing', 'Statistics', 'Modeling']}
bskills = {key: [] for key in ['Project Management', 'Consulting', 'Negotiation', 'Strategic Thinking',
                               'Business Intelligence', 'Market Analysis', 'Risk Management']}
pskills = {key: [] for key in ['Teamwork', 'Creativity', 'Communication', 'Leadership', 'Time Management', 'Adaptability']}

# Combine all skills
skills = list(sskills.keys()) + list(tskills.keys()) + list(bskills.keys()) + list(pskills.keys())


def preprocess_data(results):
    """
    Preprocess the dataset to fill missing values and standardize key columns.
    """
    # Ensure required columns exist
    required_columns = ['Descriptions', 'Company', 'Location', 'Date', 'Title']
    for col in required_columns:
        if col not in results.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset.")

    # Fill missing values
    results['Descriptions'] = results['Descriptions'].fillna('')
    results['Company'] = results['Company'].fillna('Unknown Company')
    results['Location'] = results['Location'].fillna('Unknown Location')
    results['Title'] = results['Title'].fillna('Unknown Title')

    return results

def extract_features_for_kmeans(results):
    """
    Extract numerical features suitable for K-Means clustering.
    """
    features = {}

    for skill in skills:
        # Frequency of skill mentions
        frequency = results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False).sum()

        # Average education level
        education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
        avg_education_level = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Descriptions'
        ].apply(lambda x: sum(education_mapping[edu] for edu in education_mapping if edu in x)).mean()

        # Percentage of remote jobs
        remote_keywords = r'remote|work from home|telecommute|flexible location|virtual'
        pct_remote = (
            results.loc[
                results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False) &
                results['Location'].str.contains(remote_keywords, case=False, na=False)
            ].shape[0]
            / max(frequency, 1)
        ) * 100

        # Average required years of experience
        experience = results['Descriptions'].str.extract(r'(\d+) years of experience')
        results['Experience_Years'] = pd.to_numeric(experience[0], errors='coerce')
        avg_experience = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Experience_Years'
        ].mean()

        # Average co-occurrence of other skills
        cooccurrence = results['Descriptions'].apply(
            lambda x: sum(1 for other_skill in skills if other_skill != skill and re.search(rf'\b{other_skill}\b', x, re.IGNORECASE))
        ).mean()

        # Number of unique job titles mentioning the skill
        unique_titles = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Title'
        ].nunique()

        # Job Mention Diversity (Substitute for Avg_Description_Length)
        # Measures the variety of unique job mentions across all titles
        job_mention_diversity = len(
            results.loc[results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False), 'Title'].unique()
        )

        # Skill Popularity by Industry (New Feature)
        industries = ['healthcare', 'technology', 'finance', 'education', 'retail']
        skill_popularity_by_industry = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Descriptions'
        ].apply(lambda x: sum(1 for industry in industries if industry in x.lower())).mean()

        # Demand diversity: number of unique locations
        unique_locations = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Location'
        ].nunique()

        # Industry mentions
        avg_industry_mentions = results.loc[
            results['Descriptions'].str.contains(rf'\b{skill}\b', case=False, na=False),
            'Descriptions'
        ].apply(lambda x: sum(1 for industry in industries if industry in x.lower())).mean()

        # Store features
        features[skill] = {
            'Frequency': frequency,
            'Avg_Education_Level': avg_education_level,
            'Pct_Remote': pct_remote,
            'Avg_Experience_Years': avg_experience,
            'Cooccurrence': cooccurrence,
            'Unique_Titles': unique_titles,
            'Job_Mention_Diversity': job_mention_diversity,  # New feature
            'Skill_Popularity_By_Industry': skill_popularity_by_industry,
            'Unique_Locations': unique_locations,
            'Industry_Mentions': avg_industry_mentions
        }

    # Convert features to DataFrame
    features_df = pd.DataFrame.from_dict(features, orient='index')

    # Handle missing values
    features_df.fillna(features_df.median(), inplace=True)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features_df),
        columns=features_df.columns,
        index=features_df.index
    )

    return scaled_features





# %%

# Example Usage
results = preprocess_data(results)
skill_features_df = extract_features_for_kmeans(results)

# Display standardized features
print(skill_features_df)


# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def determine_optimal_k(features_df, max_k=10):
    """
    Use the Elbow Method to determine the optimal number of clusters for K-means.
    """
    distortions = []
    silhouette_scores = []

    # Iterate through different numbers of clusters
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_df)

        # Distortion: Sum of squared distances of samples to their closest cluster center
        distortions.append(kmeans.inertia_)


    # Plot the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.grid(True)
    plt.show()

    return distortions, silhouette_scores

def perform_kmeans(features_df, n_clusters):
    """
    Perform K-means clustering with the specified number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['Cluster'] = kmeans.fit_predict(features_df)

    # Visualize clusters in 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features_df.drop('Cluster', axis=1))

    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=features_df['Cluster'], cmap='viridis', s=50)
    plt.title(f'K-means Clustering (k={n_clusters})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()

    return features_df

# Step 1: Visualize Elbow Method and Silhouette Scores
distortions, silhouette_scores = determine_optimal_k(skill_features_df)

# Step 2: Choose optimal k )
optimal_k = 4

# Step 3: Perform K-means clustering
clustered_df = perform_kmeans(skill_features_df, n_clusters=optimal_k)

# Display the clustered dataset
print(clustered_df)


# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_kmeans(features_df, n_clusters):
    """
    Perform K-means clustering with the specified number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_df)

    # Add cluster labels to the DataFrame
    clustered_df = features_df.copy()
    clustered_df['Cluster'] = labels

    return clustered_df, labels

def reduce_dimensionality_pca(features_df, n_components=2):
    """
    Reduce dimensions of data using PCA.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(features_df)
    return reduced_data

def plot_clusters(reduced_data, labels, title='K-means Clustering with PCA'):
    """
    Plot clusters using reduced dimensional data.
    """
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=labels, cmap='viridis', s=50, alpha=0.8, edgecolor='k'
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)
    plt.show()

#  usage:

clustered_df, cluster_labels = perform_kmeans(skill_features_df, n_clusters=optimal_k)

reduced_data = reduce_dimensionality_pca(skill_features_df.drop('Cluster', axis=1))

plot_clusters(reduced_data, cluster_labels)

# %%
K_mean_clustering_curriculum_sequence = {
    "Semester 1": [
        {"Course": "Programming for Applied AI", "Topics": ["Python", "SQL", "MATLAB"]},
        {"Course": "Statistics and Advanced Analytics", "Topics": ["Statistics", "Modeling", "Optimization", "NLP"]},
    ],
    "Semester 2": [
        {"Course": "Data Visualization and Storytelling", "Topics": ["Tableau", "Power BI", "Data Storytelling"]},
        {"Course": "Big Data and Cloud Computing", "Topics": ["Hadoop", "Spark", "Cloud Computing"]},
    ],
    "Semester 3": [
        {"Course": "Advanced Machine Learning and AI", "Topics": ["TensorFlow", "PyTorch", "Deep Learning"]},
        {"Course": "Leadership and Strategic Management", "Topics": ["Leadership", "Project Management", "Strategic Thinking"]},
    ],
    "Semester 4": [
        {"Course": "Business Intelligence and Consulting", "Topics": ["Business Intelligence", "Consulting", "Strategy Implementation"]},
        {"Course": "Ethics and Governance in AI", "Topics": ["Ethical AI", "Data Privacy", "Regulatory Compliance"]},
        {"Course": "Capstone Project", "Topics": ["Integrated Application of All Skills"]},
    ],
}


# %% [markdown]
# ### Attempt to combine 

# %%
import numpy as np
from sklearn.cluster import SpectralClustering
import pandas as pd

# Step 1: Define Skills and Cluster Results
skills = [
    'Data Management', 'Business Intelligence', 'Data Visualization',
    'Statistics', 'Machine Learning', 'Modeling', 'Data Analysis',
    'TensorFlow', 'PyTorch', 'Microsoft Excel', 'Cloud Computing',
    'NLP', 'Matlab', 'C++', 'Kotlin', 'Market Analysis', 
    'Strategic Thinking', 'Adaptability', 'Time Management', 
    'Negotiation', 'Risk Management', 'Java', 'Big Data', 'Teamwork',
    'Creativity', 'Project Management', 'Consulting', 'Python', 'SQL',
    'Leadership', 'Communication'
]

# Hierarchical Clustering Results
hierarchical_clusters = {
    1: ['Data Management', 'Business Intelligence'],
    2: ['Data Visualization', 'Statistics', 'Machine Learning', 'Modeling'],
    3: ['Data Analysis'],
    4: [
        'TensorFlow', 'PyTorch', 'Microsoft Excel', 'Cloud Computing', 'NLP', 
        'Matlab', 'C++', 'Kotlin', 'Market Analysis', 'Strategic Thinking', 
        'Adaptability', 'Time Management', 'Negotiation', 'Risk Management', 
        'Java', 'Big Data', 'Teamwork', 'Creativity'
    ],
    5: ['Project Management', 'Consulting'],
    6: ['Python', 'SQL'],
    7: ['Leadership'],
    8: ['Communication']
}

# K-means Clustering Results
kmeans_clusters = {
    0: ['Python'],
    1: ['Matlab', 'Kotlin', 'Java', 'C++'],
    2: ['TensorFlow', 'PyTorch'],
    3: ['Data Analysis', 'Data Visualization'],
    4: ['Leadership', 'Communication']
}

# Step 2: Initialize Co-association Matrix
n = len(skills)
skill_to_index = {skill: idx for idx, skill in enumerate(skills)}
co_association_matrix = np.zeros((n, n))

# Update matrix based on hierarchical clusters
for cluster, skill_list in hierarchical_clusters.items():
    indices = [skill_to_index[skill] for skill in skill_list]
    for i in indices:
        for j in indices:
            co_association_matrix[i][j] += 1

# Update matrix based on K-means clusters
for cluster, skill_list in kmeans_clusters.items():
    indices = [skill_to_index[skill] for skill in skill_list]
    for i in indices:
        for j in indices:
            co_association_matrix[i][j] += 1

# Normalize the matrix
co_association_matrix /= co_association_matrix.max()

# Step 3: Perform Spectral Clustering
num_clusters = 5  # Set the number of clusters
spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
final_labels = spectral.fit_predict(co_association_matrix)

# Step 4: Map Labels to Skills
final_clusters = {i: [] for i in range(num_clusters)}
for skill, label in zip(skills, final_labels):
    final_clusters[label].append(skill)

# Convert to a DataFrame for better visualization
final_clusters_df = pd.DataFrame.from_dict(final_clusters, orient='index').transpose()
print(final_clusters_df)



