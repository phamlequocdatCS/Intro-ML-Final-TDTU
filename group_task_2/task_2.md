## Task 2: 

Dataset used: [COVID-19 Dataset by MEIR NIZRI](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)

### Columns

|ID | Field                     | Description |
|---|---------------------------|-------------|
|1  | **USMER**                 | Indicates whether the patient treated medical units of the first, second or third level.      |
|2  | **MEDICAL_UNIT**          | Type of institution of the National Health System that provided the care.                     |
|3  | **SEX**                   | 1 - female. 2 - male                                                                          |
|4  | **PATIENT_TYPE**          | Type of care the patient received in the unit. 1 for returned home and 2 for hospitalization. |
|5  | **DATE_DIED**             | If the patient died indicate the date of death, and '9999-99-99' otherwise.                   |
|6  | **INTUBED**               | Whether the patient was connected to the ventilator.                                          |
|7  | **PNEUMONIA**             | Whether the patient already have air sacs inflammation or not.                                |
|8  | **AGE**                   | Age of the patient.                                                                           |
|9  | **PREGNANT**              | Whether the patient is pregnant or not.                                                       |
|10 | **DIABETES**              | Whether the patient has diabetes or not.                                                      |
|11 | **COPD**                  | Whether the patient has Chronic obstructive pulmonary disease or not.                         |
|12 | **ASTHMA**                | Whether the patient has asthma or not.                                                        |
|13 | **INMSUPR**               | Whether the patient is immunosuppressed or not.                                               |
|14 | **HIPERTENSION**          | Whether the patient has hypertension or not.                                                  |
|15 | **OTHER_DISEASE**         | Whether the patient has other disease or not.                                                 |
|16 | **CARDIOVASCULAR**        | Whether the patient has heart or blood vessels related disease.                               |
|17 | **OBESITY**               | Whether the patient is obese or not.                                                          |
|18 | **RENAL_CHRONIC**         | Whether the patient has chronic renal disease or not.                                         |
|19 | **TOBACCO**               | Whether the patient is a tobacco user.                                                        |
|20 | **CLASIFFICATION_FINAL**  | Covid test results. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or |1that the test is inconclusive.                       |
|21 | **ICU**                   | Whether the patient had been admitted to an Intensive Care Unit.                              |

For our analysis, we will discard **USMER**, **MEDICAL_UNIT**, **PATIENT_TYPE**.





1. Perform statistical analysis on the data, create graphs to understand the data. Investigate the role of each feature on the problem's objective.

2. Apply basic machine learning models to solve this problem, including Ensemble Learning methods.

3. Use models of Feed Forward Neural Network and Recurrent Neural Network (or similar models) to solve this problem.

4. Implement techniques to prevent Overfitting in the models from task (2) and task (3).

5. Once the model training is completed, please analyze the obtained result of one model (you can choose yourself). How to improve the accuracy of these implemented models.