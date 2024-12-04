
# Classification of Obesity Levels Based on Eating Habits and Physical Condition Using Data Analysis

## Project Overview

In this project we use machine learning techniques to predict obesity levels based on various factors (such as age, gender, height, and weight) and lifestyle habits (e.g. eating patterns, exercise, smoking, and water intake).  To do so we analyze the dataset titled  "Estimation of Obesity Levels Based On Eating Habits and Physical Condition” (https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), which
contains 16 features and 2111 observations.  

## Applications for Results

The results of this project could assist various organizations in enhancing their decision-making processes. Below is a summary of key organizations and their potential applications.

### 1. Public Health Organizations: 
By determining which features in the dataset are the greatest predictors of obesity levels, public health professionals could craft educational campaigns focusing on the most impactful aspects. This would provide insight into what individuals should focus on to reduce their risk of obesity.  

### 2. Health Care Providers and Practitioners:  
The results could be used by health care professionals to monitor and manage obesity. More specifically it could be applied to help create a health recommendation system by leveraging the identified key variables related to lifestyle habits, dietary patterns, and physical conditions.  Such a system could aid in identifying at risk individuals who could then be offered interventions and support.  

### 3. Insurance Companies
The analysis could help in designing custom insurance policies or health premiums based on the identified obesity risks.

## Analysis Goals

- **Leveraging Feature Importance for Public Awareness on Key Factors to Combat Obesity** To create public awareness and highlight the key factors influencing obesity, we use feature importance analysis. 

- **Analyzing the Impact of Family History and Habits** This dataset includes features such as family history of being overweight and consumption patterns, which we analyze here to determine their impact on obesity levels.

- **Predicting user engagement with technological devices** Analyze how technology usage (e.g. smartphone, TV, videogames, computers and other digital tools) correlates with obesity across various age groups. 


## Libraries and Frameworks

This project will be performed using Python. For the library list please see the 'requirements.txt' file.


## Dataset Information
This dataset is **synthetic**, and all classes are balanced, so class imbalance is not an issue. **77% of the data** was generated synthetically using the **Weka tool** and the **SMOTE filter**, while **23% of the data** was collected directly from users through a web platform.

In this project, our goal is to predict obesity levels based on various factors using a dataset from UCI. The target variable, **`NObeyesdad`**, represents obesity levels and includes **7 classes**, making this a **multi-class classification problem**. The classes are as follows:

- **Class 0: Insufficient Weight**
- **Class 1: Normal Weight**
- **Class 2: Overweight Level I**
- **Class 3: Overweight Level II**
- **Class 4: Obesity Type I**
- **Class 5: Obesity Type II**
- **Class 6: Obesity Type III**

There are 16 features in the data set, which have been renmaed for readability. The features include the following :

      - Gender                 
      - Age                       
      - Height                      
      - Weight                       
      - Family_History   (whether there is family history of obesity)          
      - High_Cal_Foods_Frequently  (does the individual consume high calorie foods frequently)  
      - Freq_Veg  (number of servings of vegetables per day)              
      - Num_Meals (number of main meals consumed per day)           
      - Snacking  (frequency of snacking)            
      - Smoke  (whether the individual is a smoker)                    
      - Water_Intake  (amount of water consumed per day)            
      - Calorie_Monitoring (whether calorie intake is being monitored)     
      - Phys_Activity  (time spent doing physical activity per week)       
      - Tech_Use  (time spent using technology per week)         
      - Freq_Alcohol  (frequency of alcohol consumption)         
      - Transportation  (type of transportaion most used: public, car, walking, motorbike, bike)        
      - Obesity_Level               


## Methodology  Outline

The following can be found in the **Notebooks** folder.

### 1. **Exploratory Data Analysis (EDA)** - see Obesity_estimation_eda.ipynb 
   - Examination of  **class distribution** and **age distribution**.
   - Identification of **outliers** and check for **missing values**.
   - Understanding the effect of specific features on obesity, such as **eating habits** and **activity levels**.
   - Analysis of **correlations** between variables.

### 2. **Data Cleaning** - see Obesity_estimation_eda.ipynb 
   - Before modeling, we performed the following data cleaning steps:
     - **Removal of duplicates**.
     - **Handling of outliers**.
     - Checking for missing values.

### 3. **One-Hot Encoding for Categorical Variables** -see Obesity_estimation_eda.ipynb 
   - Preparation of **categorical variables** for machine learning using **one-hot encoding**.
   - Scaling of numerical features using Standard Scaler.

### 4. **Machine Learning Modeling**
   - Implementation of multiple classification algorithms:
     - **Decision Tree**
     - **Random Forest**
     - **KNeighbors**
     - **XGBoost**
   - Evaluatation each model using metrics like **accuracy**, **precision**, **recall**, **F1 score**, and **log loss**.
   - **Compare model performances** to select the best-performing model using GridSearch.

### 5. **Feature Engineering**
   - We used **XGBoost, Random Forest** and **SHAP** to assess **feature importance**.
   - Utilizatiion of  **correlation matrix** insight to eliminate redundant features.

### 6. **Feature Elimination and Model Comparison**
   - Reduction of the feature set based on **feature importance** and **correlations**.
   - Comparison of performance of the models with a reduced feature set against the **baseline model**.

### 7. **Findings and Conclusion**
   - Summarization of key findings from **EDA** and **ML analysis**.
   - Determine the influence of various features in predicting obesity levels.
   - Reccomendations for future research.


### Key Observations

The correlation matrix findings show

XGBoost was the highest performing model with Random Forest performing slightly behind it. Eliminating 'Height' and 'Weight' from the features to reduce bias (as these are used to calculate BMI, a direct measure of obesity) we saw only a slight drop in performance. This demonstrates the strength of the models in predicting obesity levels from the other features. A reduced feature set can be a viable option for faster inference and simpler deployment without substantial loss of accuracy.

The results from the SHAP analysis run on models excluding 'Height' and 'Weight' showed the following features to be the top predictors of obesity levels:

      - Age
      - Frequency of Vegetables (Freq_Veg)
      - Gender
      - Water Intake,
      - Physical Activity
      - Tech Use
      - Number of Meals (Num_Meals)
      - Family History of Obesity (Yes)

The correlation matrix findings show

### Results

The table below show the performance results of the various models tested. Models run with 'Height' and 'Weight' eliminated are noted as _EWH.


![image.png](attachment:image.png)

 

### Future Scope and Next Steps

On further model applications ensure the inclusion of key features such as `Weight`, `Height`, `Age`, and `Freq_Veg`. Avoid removing these features unless computational or data collection constraints require it. XGBoost is the most robust and reliable model for this dataset when using all features. It should be considered as the primary model for deployment or further analysis.

Future experiments could include fine-tuning XGBoost hyperparameters and evaluating its performance on unseen test data or under real-world conditions.


### Team members 
[**`Arezoo khalili`**](https://github.com/Arezookhalili), [**Jyoti Narang**](https://github.com/drop2jyoti) , [**Kathryn Vozoris**](https://github.com/KathrynVozoris), [**Zekiye Erdem**](https://github.com/zekiyerdem)

### Project 
https://github.com/users/drop2jyoti/projects/2

