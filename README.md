
# Classification of Obesity Levels Based on Eating Habits and Physical Condition Using Data Analysis

## Project Overview

In this project we use machine learning techniques to predict obesity levels based on various factors (such as age, gender, height, and weight) and lifestyle habits (e.g. eating patterns, exercise, smoking, and water intake).  To do so we analyze the dataset titled  "Estimation of Obesity Levels Based On Eating Habits and Physical Condition” (https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).  The dataset
contains 16 features and 2111 observations.  

### Team members 
[**`Arezoo khalili`**](https://github.com/Arezookhalili), [**Jyoti Narang**](https://github.com/drop2jyoti) , [**Kathryn Vozoris**](https://github.com/KathrynVozoris), [**Zekiye Erdem**](https://github.com/zekiyerdem)


## Potential Applications for Results

The results of this project could assist various organizations in enhancing their decision-making processes. Below is a summary of key organizations and their potential applications. We use feature importance analysis to highlight the key factors influencing obesity.

#### - Public Health Organizations: 
By determining which features in the dataset are the greatest predictors of obesity levels, public health professionals could craft educational campaigns focusing on the most impactful aspects. This would provide insight into what individuals should focus on to reduce their risk of obesity.  

#### - Health Care Providers and Practitioners:  
The results could be used by health care professionals to monitor and manage obesity. More specifically it could be applied to help create a health recommendation system by leveraging the identified key variables related to lifestyle habits, dietary patterns, and physical conditions.  Such a system could aid in identifying at risk individuals who could then be offered interventions and support.  

#### - Insurance Companies
The analysis could help in designing custom insurance policies or health premiums based on the identified obesity risks.


- **Analyzing the Impact of Family History and Habits** This dataset includes features such as family history of being overweight and consumption patterns, which we analyze here to determine their impact on obesity levels.

- **Predicting user engagement with technological devices** Analyze how technology usage (e.g. smartphone, TV, videogames, computers and other digital tools) correlates with obesity across various age groups. 


## Libraries and Frameworks

This project will be performed using Python and Python Libraries including: 
      - Pandas
      - Numpy
      - MatPlotlib
      - ScikitLearn
      - Seaborn
      - 
For more details please see the 'requirements.txt' file.

## Dataset Information
This dataset is **synthetic**, and all classes are balanced, so class imbalance is not an issue. **77% of the data** was generated synthetically using the **Weka tool** and the **SMOTE filter**, while **23% of the data** was collected directly from users through a web platform.

The target variable, **Obesity Level**, represents obesity levels and includes **7 classes**, making this a **multi-class classification problem**. The classes are as follows:

- Class 0: Insufficient Weight
- Class 1: Normal Weight
- Class 2: Overweight Level I
- Class 3: Overweight Level II
- Class 4: Obesity Type I
- Class 5: Obesity Type II
- Class 6: Obesity Type III

There are 16 features in the data set, which have been renamed for readability:

      - Gender                 
      - Age                       
      - Height                      
      - Weight                       
      - Family_History   (whether there is family history of obesity)          
      - High_Cal_Foods_Frequently   (frequency of consumption of high calorie foods)  
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


## Methodology Outline

The following can be found in the **Notebooks** folder.

### A.  Exploratory Data Analysis

      Examination of  **class distribution** and **age distribution**.
      Identification of **outliers** and  **missing values**.
      Examination of the effect of specific features on obesity, such as **eating habits** and **activity levels**.
      Analysis of **correlations** between variables.

   #### 1. **Data Cleaning**  
       - Before modeling, we performed the following data cleaning steps:
       - **Removal of duplicates**.
       - **Handling of outliers**.
       - **Checking for missing values.**

   #### 2. Visualizations and Observations
     - Box plots, scatter plots, bar plots, Histograms 
     - Correlation analysis
     - Conclusions and summary of findings

   ### 3. Key Observations 

   We have highest number of people with Obesity_Type I
   On average most people have family history Suof obsesity.
   Most people use some form of vehicles while only ~2.7% prefers walking/using bike That's concerning!
   More female are Obese as compared to male.
   We noticed a relatively strong positive correlation between the Weight and Height.
   Outliers are present in Age.



### B. Machine Learning Analysis:

#### 1/. **Preprocessing ** 
   - Preparation of **categorical variables** for machine learning using **one-hot encoding**.
   - Scaling of numerical features using Standard Scaler.
   - Label encoding of target variable.

#### 2.. **Machine Learning Modeling**  
   - Implementation of multiple classification algorithms:
     - **Decision Tree**
     - **Random Forest**
     - **KNeighbors**
     - **XGBoost**
   - Evaluation of each model using the following metrics: **accuracy**, **precision**, **recall**, and **F1 score**
   

#### 3. Hyperparameter Tuning
      - Tuning of hyperparamter tuning on XgBoost and Random Forest
      - **Comparison of model performances** to select the best-performing model using **GridSearch**.
      - Applied 5 fold cross validation 

#### 4. **Feature Engineering**
   - We used **XGBoost, Random Forest** and **SHAP** to assess **feature importance**.
   - Utilizatiion of  **correlation matrix** insight to eliminate redundant features.

#### 5. **Feature Elimination and Model Comparison** 
   - Reduction of the feature set based on **feature importance** and **correlations**.
   - Comparison of performance of the models with a reduced feature set against the **baseline model**.


#### 5. **Findings and Conclusion**  
   - Summarization of key findings from **EDA** and **ML analysis**.
   - Determine the influence of various features in predicting obesity levels.
   - Reccomendations for future research.


### Key Observations

XGBoost was the highest performing model with Random Forest performing slightly behind it. Eliminating 'Height' and 'Weight' from the features to reduce bias (as these are used to calculate BMI, a direct measure of obesity) we saw only a slight drop in performance. This demonstrates the strength of the models in predicting obesity levels from the other features. A reduced feature set can be a viable option for faster inference and simpler deployment without substantial loss of accuracy.


The results from the **SHAP** analysis run on models excluding 'Height' and 'Weight' showed the following features to be the top predictors of obesity levels:

      - Age
      - Frequency of Vegetables (Freq_Veg)
      - Gender
      - Water Intake
      - Physical Activity
      - Tech Use
      - Number of Meals (Num_Meals)
      - Family History of Obesity (Yes)

For higher levels of obesity (Obesity levels 1, 2, and 3) the following features are the main indicators in predicting obesity:

      - Gender
      - Age
      - Water Intake
      - Tech Use

Therefore these are the features of primary importance when assessing an indivuals risk of obesity. The features could be considered by institutions and public health departments looking to reduce obesity levels across populations.

### Results

The table below shows the performance results of the various models tested. Models run with the features 'Height' and 'Weight' eliminated are noted as _EWH.


![Model Comparison Table](reports/image1.png)

 
### Future Scope and Next Steps

XGBoost is the most robust and reliable model for this dataset. It should be considered as the primary model for deployment or further analysis. Future experiments could include fine-tuning XGBoost hyperparameters and evaluating its performance on unseen test data or under real-world conditions. On future model applications ensure the inclusion of key features such as `Weight`, `Height`, `Age`, and `Freq_Veg`. Avoid removing these features unless computational or data collection constraints require it.  We considered the miscategorized data using a **Confusion matrix**.  These misclassified labels could also be further investigated.

#### Task Allocation

#### 1. **Exploratory Data Analysis (EDA)**
   - **Lead:** Jyoti
   - **Support:** Arezoo
   - Tasks: Examine class and age distributions, identify outliers, explore missing values, and analyze correlations.

#### 2. **Data Cleaning**
   - **Lead:** Arezoo
   - **Support:** Zekiye
   - Tasks: Remove duplicates, handle outliers, and manage missing values as necessary.

#### 3. **One-Hot Encoding for Categorical Variables**
   - **Lead:** Kathryn
   - Tasks: Prepare categorical variables for machine learning through one-hot encoding.

#### 4. **Machine Learning Modeling**
   - **Lead:** Zekiye
   - **Support:** Jyoti
   - Tasks: Implement Decision Tree, Random Forest, Logistic Regression, and Naive Bayes models. Evaluate and compare models on metrics (accuracy, precision, recall, F1 score, log loss).

#### 5. **Feature Engineering**
   - **Lead:** Zekiye
   - **Support:** Arezoo
   - Tasks: Use Random Forest for feature importance assessment, and analyze correlation matrix to remove redundant features.

#### 6. **Feature Elimination and Model Comparison**
   - **Lead:** Jyoti
   - **Support:** Kathryn
   - Tasks: Perform feature elimination, and compare model performance with a reduced feature set vs. the baseline model.

#### 7. **Findings and Conclusion**
   - **Lead:** Kathryn
   - **Support:** All Members
   - Tasks: Summarize key findings, draw conclusions, and compile the final report.


### Project 
https://github.com/users/drop2jyoti/projects/2

[Appendix]()
