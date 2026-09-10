import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

def create_project_report():
    doc = Document()
    
    # Title
    title = doc.add_heading('Predictive Maintenance of Tool Wear in CNC Machines', 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    
    # Overview
    doc.add_heading('1. Project Overview', level=1)
    doc.add_paragraph(
        "This project develops a Predictive Maintenance System for CNC (Computer Numerical Control) "
        "machine tools using Machine Learning. The system analyses sensor data to predict both "
        "tool failure (Binary Classification) and remaining tool life (Regression). It provides "
        "a web-based dashboard with real-time predictions, batch predictions, and maintenance recommendations."
    )
    
    # Dataset
    doc.add_heading('2. Dataset details', level=1)
    doc.add_paragraph(
        "The project uses the 'AI4I 2020 Predictive Maintenance Dataset' sourced from the "
        "UCI Machine Learning Repository."
    )
    doc.add_paragraph("Dataset Properties:", style='List Bullet')
    doc.add_paragraph("Total Rows: 10,000", style='List Bullet 2')
    doc.add_paragraph("Total Features: 11 input features + 1 target feature", style='List Bullet 2')
    doc.add_paragraph("Failure Rate: ~3.4%", style='List Bullet 2')
    
    # Features
    doc.add_heading('3. Features Used in the Project', level=1)
    doc.add_heading('3.1 Raw Features', level=2)
    doc.add_paragraph("Type — Machine quality grade (L/M/H)", style='List Bullet')
    doc.add_paragraph("Air temperature — Ambient temperature (K)", style='List Bullet')
    doc.add_paragraph("Process temperature — Machining temperature (K)", style='List Bullet')
    doc.add_paragraph("Rotational speed — Spindle speed (RPM)", style='List Bullet')
    doc.add_paragraph("Torque — Cutting torque (Nm)", style='List Bullet')
    doc.add_paragraph("Tool wear — Accumulated wear time (min)", style='List Bullet')
    
    doc.add_heading('3.2 Engineered Features', level=2)
    doc.add_paragraph("Power — Mechanical power (W)", style='List Bullet')
    doc.add_paragraph("Temp_diff — Temperature differential (K)", style='List Bullet')
    doc.add_paragraph("Torque_speed_ratio", style='List Bullet')
    doc.add_paragraph("Wear_rate", style='List Bullet')
    
    # Machine Learning Models
    doc.add_heading('4. Models and Algorithms', level=1)
    doc.add_paragraph(
        "The system trains and evaluates multiple machine learning algorithms to select the best performer."
    )
    doc.add_heading('4.1 Classification Models (Failure Prediction)', level=2)
    doc.add_paragraph("Decision Tree: Interpretable baseline model.", style='List Bullet')
    doc.add_paragraph("Random Forest: 200 estimators, class weights balanced.", style='List Bullet')
    doc.add_paragraph("Gradient Boosting: 150 estimators, learning rate = 0.1.", style='List Bullet')
    doc.add_paragraph("XGBoost: Optimised with scale_pos_weight to handle class imbalance.", style='List Bullet')
    doc.add_paragraph("Extra Trees: Extremely randomised trees.", style='List Bullet')
    doc.add_paragraph("Support Vector Machine (SVM): RBF Kernel with C=10, class weights balanced.", style='List Bullet')
    doc.add_paragraph("Logistic Regression: L2 penalty, max iterations = 1000.", style='List Bullet')
    
    doc.add_heading('4.2 Regression Model (Remaining Useful Life)', level=2)
    doc.add_paragraph("Random Forest Regressor: 200 estimators for wear estimation.", style='List Bullet')
    
    # Training and Testing Sets
    doc.add_heading('5. Training and Testing Split', level=1)
    doc.add_paragraph(
        "The dataset is divided into training and testing sets to evaluate model generalization."
    )
    doc.add_paragraph("Training Set: 80% (8,000 samples). Used to train the models.", style='List Bullet')
    doc.add_paragraph("Testing Set: 20% (2,000 samples). Used to evaluate the models.", style='List Bullet')
    doc.add_paragraph(
        "The split uses 'stratification' on the target class (Machine failure) to ensure that the "
        "training and testing sets have the same proportion of failures and normal conditions."
    )
    
    # Project Flow
    doc.add_heading('6. Full Project Flow', level=1)
    flow = [
        "Data Loading: Load the AI4I 2020 dataset from the CSV file.",
        "Data Cleaning: Remove duplicate rows, drop completely empty rows, forward-fill/backward-fill missing values, and remove non-feature identifier columns.",
        "Label Encoding: Encode the categorical 'Type' feature (L, M, H) into integer values using a LabelEncoder.",
        "Feature Engineering: Generate new physics-based features such as Power, Temperature Differential, Torque-Speed Ratio, and Wear Rate from raw sensor readings.",
        "Feature Selection: Separate the input features (X) and target outputs (y_classification for failure, y_regression for tool wear).",
        "Feature Scaling: Standardize the features using a StandardScaler so they all have a mean of 0 and a variance of 1.",
        "Data Splitting: Split the preprocessed data into 80% training and 20% testing sets using stratified sampling.",
        "Model Training: Train 7 different classification models and 1 regression model using the training set.",
        "Model Evaluation: Evaluate all models on the testing set using metrics like Accuracy, F1-Score, and ROC-AUC.",
        "Model Selection & Saving: Automatically identify the best performing classification model based on the F1-Score, and save it along with the regressor, scaler, and encoder to disk.",
        "Inference/Dashboard: Launch the Streamlit web dashboard to provide real-time prediction and interactive visualizations to operators."
    ]
    
    for i, step in enumerate(flow, 1):
        doc.add_paragraph(f"{i}. {step}")
        
    doc.save(r'c:\Users\jesuschrist\Downloads\Major_Project\Project_Details_Report.docx')
    print("Document saved successfully.")

if __name__ == '__main__':
    create_project_report()
