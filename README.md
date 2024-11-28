
# Classification of Obesity Levels Based on Eating Habits and Physical Condition Using Data Analysis

## Project Overview

Our aim in this project is to create a machine-learning model based on demographic features (such as age, gender, height, and weight) and lifestyle habits (e.g. eating patterns, exercise, 
smoking, and water intake) for predicting obesity levels. To do so we analyze the dataset titled  "Estimation of Obesity Levels Based On Eating Habits and Physical Condition” (https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) which
contains 16 features and 2111 observations.

## Potential Business Cases in Different Areas

The results of this project can assist various organizations in enhancing their decision-making processes. We have provided a concise summary of key organizations and their potential applications.

### 1. Public Health Organizations: 
By determining which features in the dataset are the greatest predictors of obesity levels, public health professionals could craft educational campaigns focusing on the most impactful aspects. This would provide insight into what individuals should focus on to reduce their risk of obesity.  

### 2. Health Care Providers and Practitioners:  
The results could be used by health care professionals to monitor and manage obesity. More specifically it could be applied to help create a health recommendation system by leveraging the identified key variables related to lifestyle habits, dietary patterns, and physical conditions.  Such a system could aid in identifying at risk individuals who could  then be offered interventions and support.  

### 3. Insurance Companies
The analysis can help in designing custom insurance policies or health premiums based on the identified obesity risks.

## Analysis Goals

- **Leveraging Feature Importance for Public Awareness on Key Factors to Combat Obesity** To create public awareness and highlight the key factors influencing obesity, we  will use feature importance analysis. 

- **Analyzing the Impact of Family History and Habits** This dataset includes features such as family history of overweight and consumption patterns, which will be analyzed to study their impact on obesity levels.

- **Predicting user engagement with technological devices** Determine how technology usage (e.g. smartphone, TV, videogames, computers and other digital tools) correlates with obesity across various age group. 


## Libraries and Frameworks

This project will perform by using Python. For the library list please see the 'requirements.txt' file.

## Methodology

In this project, our goal is to predict obesity levels based on various factors using a dataset from UCI. The target variable, **`NObeyesdad`**, represents obesity levels and includes **7 classes**, making this a **multi-class classification problem**. The classes are as follows:

- **Insufficient Weight**
- **Normal Weight**
- **Overweight Level I**
- **Overweight Level II**
- **Obesity Type I**
- **Obesity Type II**
- **Obesity Type III**

Below is the methodology we followed.

### Dataset Information
This dataset is **synthetic**, and all classes are balanced, so class imbalance is not an issue. **77% of the data** was generated synthetically using the **Weka tool** and the **SMOTE filter**, while **23% of the data** was collected directly from users through a web platform.

### 1. **Exploratory Data Analysis (EDA)**
   - Examine **class distribution** and **age distribution**.
   - Identify **outliers** and explore **missing values**.
   - Understand the effect of specific features on obesity, such as **eating habits** and **activity levels**.
   - Analyze **correlations** between variables.

### 2. **Data Cleaning**
   - Before modeling, perform the following data cleaning steps:
     - **Remove duplicates**.
     - **Handle outliers** (if any).
     - Address **missing values**.

### 3. **One-Hot Encoding for Categorical Variables**
   - Prepare **categorical variables** for machine learning by applying **one-hot encoding**.

### 4. **Machine Learning Modeling**
   - Implement multiple classification algorithms:
     - **Decision Tree**
     - **Random Forest**
     - **kNN**
     - **XGBoost**
   - Evaluate each model using metrics like **accuracy**, **precision**, **recall**, **F1 score**, and **log loss**.
   - **Compare model performances** to select the best-performing model.

### 5. **Feature Engineering**
   - Use **Random Forest** or **SHAP** to assess the **feature importance**.
   - Utilize **correlation matrix** insights to eliminate redundant features.

### 6. **Feature Elimination and Model Comparison**
   - Reduce the feature set based on **feature importance** and **correlations**.
   - Compare performance of models with the reduced feature set against the **baseline model**.

### 7. **Findings and Conclusion**
   - Summarize key findings from **EDA** and **ML analysis**.
   - Draw conclusions on the influence of various features in predicting obesity levels.

## Workload Distribution


#### Group Members
- **Jyoti**
- **Arezoo**
- **Zekiye**
- **Kathryn**

#### Task Allocation

#### 1. **Exploratory Data Analysis (EDA)** - Done
   - **Lead:** Jyoti
   - **Support:** Arezoo
   - Tasks: Examine class and age distributions, identify outliers, explore missing values, and analyze correlations.

#### 2. **Data Cleaning** - Done
   - **Lead:** Arezoo
   - **Support:** Zekiye
   - Tasks: Remove duplicates, handle outliers, and manage missing values as necessary.

#### 3. **One-Hot Encoding for Categorical Variables** - Done
   - **Lead:** Kathryn
   - Tasks: Prepare categorical variables for machine learning through one-hot encoding.

#### 4. **Machine Learning Modeling** - Done
   - **Lead:** Zekiye
   - **Support:** Jyoti
   - Tasks: Implement Decision Tree, Random Forest, Logistic Regression, and Naive Bayes models. Evaluate and compare models on metrics (accuracy, precision, recall, F1 score, log loss).

#### 5. **Feature Engineering**
   - **Lead:** Zekiye
   - **Support:** Arezoo
   - Tasks: Use Random Forest or SHAP for feature importance assessment, and analyze correlation matrix to remove redundant features.

#### 6. **Feature Elimination and Model Comparison**
   - **Lead:** Jyoti
   - **Support:** Kathryn
   - Tasks: Perform feature elimination, and compare model performance with a reduced feature set vs. the baseline model.

#### 7. **Findings and Conclusion**
   - **Lead:** Kathryn
   - **Support:** All Members
   - Tasks: Summarize key findings, draw conclusions, and compile the final report.


### Key Observations

### Results

### Future Scope and Next Steps

### Team members 
[**`Arezoo khalili`**](https://github.com/Arezookhalili), [**Jyoti Narang**](https://github.com/drop2jyoti) , [**Kathryn Vozoris**](https://github.com/KathrynVozoris), [**Zekiye Erdem**](https://github.com/zekiyerdem)

### Project 
https://github.com/users/drop2jyoti/projects/2

