from flask import Flask, request, jsonify # Importar Flask, request, jsonify para manejar solicitudes HTTP y crear una aplicación web
import joblib # Importar joblib para cargar el modelo PCA
import pandas as pd # Importar pandas para manejar datos
from sklearn.preprocessing import StandardScaler # Importar StandardScaler para escalar datos
from tensorflow.keras.models import load_model # Importar load_model para cargar el modelo de red neuronal
from flask_cors import CORS # Importar CORS para permitir solicitudes de cualquier origen

app = Flask(__name__) # Crear una aplicación web
CORS(app)  # Esto permite todas las solicitudes de cualquier origen

# Cargar los modelos PCA y red neuronal
pca_model = joblib.load('models\\pca_model.pkl')
nn_model = load_model('models\\NNmodel.h5')
scaler = joblib.load('models\\scalerred.pkl')

# Columnas numericas 
numeric_columns = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", 
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", 
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", 
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]
# Columnas Categoricas
categorical_columns = [
    'BusinessTravel', 'Department', 'Education', 'EducationField', 
    'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
    'WorkLifeBalance'
]
# Columnas esperadas al momento de hacer la dummificacion de las variables categoricas
expected_columns = [
    "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
    "Department_Human Resources", "Department_Research & Development", "Department_Sales",
    "Education_1", "Education_2", "Education_3", "Education_4", "Education_5",
    "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing",
    "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree",
    "EnvironmentSatisfaction_1", "EnvironmentSatisfaction_2", "EnvironmentSatisfaction_3", "EnvironmentSatisfaction_4",
    "Gender_Female", "Gender_Male", "JobInvolvement_1", "JobInvolvement_2", "JobInvolvement_3", "JobInvolvement_4",
    "JobLevel_1", "JobLevel_2", "JobLevel_3", "JobLevel_4", "JobLevel_5",
    "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician",
    "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", 
    "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative",
    "JobSatisfaction_1", "JobSatisfaction_2", "JobSatisfaction_3", "JobSatisfaction_4",
    "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single",
    "OverTime_No", "OverTime_Yes", "PerformanceRating_3", "PerformanceRating_4",
    "RelationshipSatisfaction_1", "RelationshipSatisfaction_2", "RelationshipSatisfaction_3", "RelationshipSatisfaction_4",
    "StockOptionLevel_0", "StockOptionLevel_1", "StockOptionLevel_2", "StockOptionLevel_3",
    "WorkLifeBalance_1", "WorkLifeBalance_2", "WorkLifeBalance_3", "WorkLifeBalance_4"
]

@app.route('/predict', methods=['POST'])
def predict():
    """ Realiza una predicción basada en datos de entrada en formato JSON.
    Los datos de entrada deben contener dos claves: 'numerical' y 'categorical', 
    que corresponden a los datos numéricos y categóricos respectivamente.
    Procesos realizados:
    1. Separación de datos numéricos y categóricos.
    2. Transformación PCA a los datos numéricos, obteniendo solo las primeras dos componentes.
    3. Creación de variables dummy para los datos categóricos.
    4. Reordenamiento de columnas para coincidir con el modelo de entrenamiento.
    5. Unión de datos transformados.
    6. Aplicación de una segunda estandarización a los datos.
    7. Realización de la predicción utilizando un modelo de red neuronal.

    Returns:
        Un objeto JSON que contiene la predicción realizada.
    """
    # Datos de entrada en formato JSON
    data = request.json  
    
    # Agrega un mensaje para depurar los datos recibidos
    # print("Datos recibidos:", data)
    
    # Separar datos numéricos y categóricos
    num_data = pd.DataFrame([data['numerical']], columns=numeric_columns)
    cat_data = pd.DataFrame([data['categorical']], columns=categorical_columns)

    # Aplicar transformación PCA a datos numéricos y obtener solo las primeras dos componentes
    num_data = StandardScaler().fit_transform(num_data) 
    num_data_pca = pca_model.transform(num_data)[:, :2]
    num_data_pca = pd.DataFrame(num_data_pca, columns=['PC1', 'PC2'])
    # Obtener las variables cualitativas 
    cat_data_dummies = pd.DataFrame(False, index=cat_data.index, columns=expected_columns)

    # Asegurarse de que todas las columnas esperadas están presentes en el DataFrame de entrada, igualando la estructura con la que se entrenó el modelo
    for index, row in cat_data.iterrows():
        for col in cat_data.columns:
            column_name = f"{col}_{row[col]}"
            if column_name in expected_columns:
                cat_data_dummies.loc[index, column_name] = True

    # Reordenar las columnas para que coincidan con el modelo de entrenamiento
    cat_data_dummies = cat_data_dummies[expected_columns]

    # Unir datos transformados
    full_data = pd.concat([pd.DataFrame(num_data_pca), cat_data_dummies], axis=1)
    
    # Aplicar una segunda escalizacion a los datos
    full_data.columns = full_data.columns.astype(str)
    full_data = scaler.transform(full_data)
    
    # Realizar predicción
    prediction_prob = nn_model.predict(full_data)  # Esto devuelve una probabilidad
    prediction_class = (prediction_prob[:, 1] >= 0.5).astype(int)  # Convierte en clase
    return jsonify({'prediction': int(prediction_class[0])})

if __name__ == '__main__':
    app.run()