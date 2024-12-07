### Dataset

The dataset "Estimation of Obesity Levels Based On Eating Habits and Physical Condition “ contains 16 features and 2111 observations. The dataset contains information on individuals from Peru, Mexico and Columbia.

Dataset Link: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

The key variables (features) in this dataset, based on feature importance and relevance to the problem of obesity, are:

•	Gender

•	Age:  A person's age may affect their metabolic rate and lifestyle, influencing obesity levels.

•	Height: Useful for calculating Body Mass Index (BMI), which is closely related to obesity.

•	Weight: One of the most critical variables for determining obesity.

•	Family History with Overweight: Whether a person has a genetic predisposition to being overweight or obese, which can significantly impact obesity risk.

•	Frequent Consumption of High-Calorie Food (FAVC): This reflects dietary habits, particularly the consumption of high-calorie foods, which is a known risk factor for obesity.

•	Frequency of Consumption of Vegetables (FCVC): A higher intake of vegetables is typically associated with a healthier diet and lower risk of obesity.

•	Number of Meals (NCP): The number of meals consumed daily can impact weight gain, especially if portion sizes are not controlled.

•	Consumption of Food Between Meals (CAEC): Eating snacks or meals outside of regular eating hours can contribute to weight gain.

•	Smoking Habit (SMOKE): Smoking status can influence metabolism and overall health, affecting obesity.

•	Water Intake (CH2O): Adequate water intake can aid in digestion and prevent overeating, indirectly influencing weight management.

•	Alcohol Consumed (CALC): Regular or frequent alcohol consumption adds extra calories, contributing to weight gain.

•	Calorie consumption monitoring (SCC): Regular or frequent calorie consumption contributes to weight gain.

•	Physical Activity Frequency (FAF): Frequency of physical activity is a key factor in energy expenditure and weight control.

•	Time Spent Using Technology (TUE): Time spent on technology or screens is often linked to sedentary behavior, which can contribute to obesity.

•	Transportation Mode (MTRANS): The mode of transportation, such as walking or using public transportation, reflects levels of physical activity in daily life.

•	Outcome variable (NOBeyesdad): Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 

#### After renaming the columns for better readability :
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
