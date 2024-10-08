# Project Title

Developing a Predictive Model in GSTN

## Description

This project offers innovative AI and ML algorithms and solutions to address the challenge presented by GSTN using the provided dataset. This initiative is part of the GSTN hackathon, which aims to promote collaboration between academia and industry experts, driving the creation of effective and insightful solutions to enhance the GST analytics framework

## Getting Started

### Dependencies

* OS - Windows 11
* IDE - Visual Studio (>1.92.2) or IntelliJ
* Python > 3.12
* Please note:  dataset not available in github due to Large file Size, hence create data folder by your own and add files for Train, Test, and Inference set as shown below

  ![image](https://github.com/user-attachments/assets/c62977a5-2f7f-4d76-8d76-c8aff34165d3)


### Installing

1. Clone the repo
   ```sh
   git clone https://github.com/Sirilkv5/GSTScience
   ```
2. Install requirements.txt packages
   ```sh
   pip install -r requirements.txt
   ```
3. Load data and ensure below code matches with the input files
  ```sh
  def load_data():
  . . .
   df_X_train = pd.read_csv('data/X_Train_Data_Input.csv')
   df_X_test = pd.read_csv('data/X_Test_Data_Input.csv')
   df_Y_train = pd.read_csv('data/Y_Train_Data_Target.csv')
   df_Y_test = pd.read_csv('data/Y_Test_Data_Target.csv')
  . . .

 def main():
 . . .
    # Make predictions on new data - a separate inference dataset
    df_X_inference = pd.read_csv('data/X_Train_Data_Input_Inference.csv')
    df_Y_inference = pd.read_csv('data/Y_Train_Data_Target_Inference.csv')
. . .
 ```

### Executing program

* Run below command
```
py new.py
```

<!-- CONTACT -->
## Contact

Sirilakshmi S  -
Linkedin - https://www.linkedin.com/in/sirilakshmi-srinivasa-59859326/ 
Email - sirilkv5@gmail.com

Project Link: https://github.com/Sirilkv5/GSTScience

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Authors

Contributors names and contact info

* Sirilakshmi - https://github.com/Sirilkv5
* Venugopala - 


## Version History

* 0.1
    * Initial Release


## Acknowledgments


* [catboost](https://catboost.ai/docs/concepts/python-reference_catboostclassifier)
* [xgboost](https://xgboost.readthedocs.io/en/stable/)
* [GSTN](https://github.com/GSTNIndia
