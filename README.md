# ML Statistical Methods for Classification and Approximation

**Author**: Hon Wa Ng\
**Date**: November 2024  

## Overview

This project aims to optimize data science curricula based on real-world job market demand. By analyzing job postings, clustering required skills, and identifying industry trends, this project provides insights into aligning educational programs with current market needs.

The dataset is sourced through web scraping, processed using NLP techniques, and analyzed through clustering algorithms to extract meaningful skill patterns.

The dataset is included in this repository under the data/ directory.

## Objectives

- Extract and process data science job postings for skill demand analysis.
- Apply text vectorization (TF-IDF, OpenAI embeddings) for feature extraction.
- Implement clustering models (K-Means, DBSCAN, Hierarchical Clustering) to group similar skills.
- Generate curriculum recommendations based on industry trends.

## Repository Structure
```bash
DATA-DRIVEN-DATA-SCIENCE-CURRICULUM-OPTIMIZATION/
│── data/                              # Dataset storage
│   ├── webscraping_results.csv         # Extracted job posting data
│
│── doc/                               # Documentation files
│   ├── project_requirement.pdf         # Original project requirement
│   ├── project_report.pdf              # Detailed project analysis
│   ├── dendrogram.png                   # Visual representation of clustering
│
│── src/                                # Source code
│   ├── Data-Science-Curriculum-Optimization.py  # Core script for analysis
│
│── LICENSE                             # License information
│── requirements.txt                     # Dependencies for running the project
```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/Data-Driven-Data-Science-Curriculum-Optimization.git
cd Data-Driven-Data-Science-Curriculum-Optimization

```

### 2. Install Dependencies
Ensure you have Python installed (>=3.7), then run:
```
pip install -r requirements.txt

```

---
## How to Run the Project
Execute the classification script:

```
python src/Data-Science-Curriculum-Optimization.py

```
The script will:

- Load and preprocess job market data.
- Apply NLP feature extraction for skill representation.
- Perform clustering analysis to identify key skill groups.
- Generate curriculum recommendations based on industry demand.

---
## Methods Used

1. Data Extraction & Preprocessing
- Web scraping job listings to extract required skills.
- Cleaning & normalizing job descriptions for NLP processing.
2. Feature Engineering
- TF-IDF vectorization for text representation.
- Embedding-based representations using OpenAI API.
3. Clustering & Analysis
- K-Means Clustering: Skill demand categorization.
- DBSCAN: Detecting core vs. outlier skills.
- Hierarchical Clustering: Generating industry skill trees.
4. Curriculum Optimization
- Mapping identified skill clusters to data science courses.
- Recommending upskilling paths based on industry trends.


---

## Results & Analysis

- Identified key skill clusters that distinguish generalists, specialists, and niche areas.
- Visualized skill relationships through dendrograms and clustering heatmaps.
- Suggested curriculum enhancements based on data-driven insights.
  
Refer to the project_report.pdf in the doc/ folder for detailed results.
---
## License
This project is licensed under the MIT License.



