import random
from flask import jsonify, request
from flask import request, jsonify
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import joblib  # Para cargar el PCA

app = Flask(__name__)

# Habilitar CORS para todas las rutas y orígenes
CORS(app, resources={r"/*": {"origins": "*"}})

# Cargar los modelos
pca_model = joblib.load('pca_model.pkl')  # Cargar el PCA entrenado
model = load_model('NNmodel.h5')  # Cargar el modelo de red neuronal

# Diccionario de mapeo para las variables categóricas
categorical_encodings = {
    'Attrition': {'Yes': 1, 'No': 0},
    'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
    'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
    'EducationField': {'Other': 0, 'Life Sciences': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5},
    'Gender': {'Male': 0, 'Female': 1},
    'JobRole': {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
                'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8},
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'OverTime': {'Yes': 1, 'No': 0}
}

expected_encoded_columns = [
    'PC1', 'PC2',
    'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
    'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
    'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'EnvironmentSatisfaction_1', 'EnvironmentSatisfaction_2', 'EnvironmentSatisfaction_3', 'EnvironmentSatisfaction_4',
    'Gender_Female', 'Gender_Male',
    'JobInvolvement_1', 'JobInvolvement_2', 'JobInvolvement_3', 'JobInvolvement_4',
    'JobLevel_1', 'JobLevel_2', 'JobLevel_3', 'JobLevel_4', 'JobLevel_5',
    'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'JobSatisfaction_1', 'JobSatisfaction_2', 'JobSatisfaction_3', 'JobSatisfaction_4',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'OverTime_No', 'OverTime_Yes',
    'PerformanceRating_3', 'PerformanceRating_4',
    'RelationshipSatisfaction_1', 'RelationshipSatisfaction_2', 'RelationshipSatisfaction_3', 'RelationshipSatisfaction_4',
    'StockOptionLevel_0', 'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
    'WorkLifeBalance_1', 'WorkLifeBalance_2', 'WorkLifeBalance_3', 'WorkLifeBalance_4'
]

# Las columnas numéricas que se usaron en el PCA
pca_columns = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]


@app.route('/apply_pca', methods=['POST'])
def apply_pca():
    # Obtener los datos del formulario (en formato JSON)
    data = request.json
    print("Datos recibidos para PCA:")
    print(data)  # Imprime los datos tal como los recibe el backend

    # Aplicar encoding en las variables categóricas
    for column, encoding in categorical_encodings.items():
        if column in data:
            value = data[column]
            # Si el valor no es válido, asignamos un valor predeterminado (por ejemplo, el primer valor válido)
            if value == '' or value is None:
                data[column] = list(encoding.keys())[0]
            else:
                # Asignar valor válido, si no existe usar el primero
                data[column] = encoding.get(value, list(encoding.keys())[0])

    # Asegurarse de que las columnas numéricas estén presentes y no estén vacías
    for col in pca_columns:
        if col not in data or data[col] == '' or data[col] is None:
            data[col] = 0  # Asignar 0 si está vacía o es None

    # Convertir los datos en un DataFrame para usar con el modelo
    df = pd.DataFrame([data])
    print(f"Dimensiones antes de PCA: {df[pca_columns].shape}")

    # Verifica si hay columnas con valores faltantes (NaN) y maneja estos casos
    if df.isnull().any().any():
        print("Hay valores nulos en los datos, se reemplazarán por 0 o valores predeterminados.")
        # Llenar valores nulos con 0 (o con algún otro valor adecuado)
        df.fillna(0, inplace=True)

    # Asegurarse de que las columnas numéricas no tengan valores vacíos
    df[pca_columns] = df[pca_columns].replace('', 0)

    # Aplicar PCA a las columnas numéricas
    pca_features = pca_model.transform(df[pca_columns])

    # Devolver las características transformadas por PCA
    # Convertir el resultado a lista para enviarlo al frontend
    pca_data = pca_features.tolist()
    return jsonify({'pca_features': pca_data})


def process_data_with_pca(employee, numeric_data, pca_columns, pca_model, cualitativas_columns):
    """
    Esta función realiza el procesamiento de datos, aplica PCA a las variables numéricas,
    y codifica las variables categóricas con One-Hot Encoding.

    Parámetros:
        employee: DataFrame con los datos del empleado (con todas las columnas).
        numeric_data: DataFrame solo con las columnas numéricas que se usarán para PCA.
        pca_columns: Lista de columnas numéricas que se usarán para PCA.
        pca_model: El modelo PCA entrenado.
        cualitativas_columns: Lista de columnas categóricas que se deben codificar.

    Retorna:
        DataFrame con las columnas PCA, las columnas categóricas codificadas, y las columnas procesadas.
    """

    # Primero, extraemos las dos primeras componentes principales con PCA
    # Trabajamos con una copia de los datos numéricos
    numeric_data2 = numeric_data.copy()

    # Escalamos los datos
    numeric_data2 = StandardScaler().fit_transform(numeric_data2)

    # Aplicamos PCA
    pca_transformed_data = pca_model.transform(
        numeric_data2)[:, :2]  # Extraemos las 2 primeras componentes
    # Creamos un DataFrame con los resultados del PCA
    pca_df = pd.DataFrame(data=pca_transformed_data, columns=['PC1', 'PC2'])

    print("Componentes principales:")
    # Muestra las primeras filas de las componentes principales
    print(pca_df.head())

    # Ahora trabajamos con las columnas categóricas
    employee_cual = employee.copy()  # Copia de los datos del empleado
    # Eliminamos las columnas numéricas de la copia
    employee_cual = employee_cual.drop(columns=pca_columns)
    # Eliminamos columnas no relevantes
    employee_cual = employee_cual.drop(
        columns=["EmployeeCount", "Over18", "EmployeeNumber"])

    # Concatenamos los datos de PCA con las columnas categóricas
    final_data = pd.concat(
        [pca_df, employee_cual.reset_index(drop=True)], axis=1)

    # Codificamos las variables categóricas usando One-Hot Encoding
    final_data = pd.get_dummies(final_data, columns=cualitativas_columns)

    print("Datos finales después de aplicar One-Hot Encoding:")
    print(final_data.head())  # Muestra las primeras filas de los datos finales

    return final_data


@app.route('/encode_categorical', methods=['POST'])
def encode_categorical(categorical_data):
    # Recibir los datos categóricos
    data = request.json

    # Convertir los datos en un DataFrame
    categorical_data = pd.DataFrame([data])
    print("Datos recibidos para codificación:")
    print(categorical_data)  # Imprime los datos recibidos para depuración

    # Codificar las variables categóricas con pd.get_dummies
    categorical_encoded = pd.get_dummies(categorical_data)

    # Asegurarse de que todas las columnas esperadas estén presentes
    expected_encoded_columns = [
        'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
        'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
        'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
        'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing',
        'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
        'Gender_Female', 'Gender_Male', 'JobRole_Healthcare Representative', 'JobRole_Human Resources',
        'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director',
        'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
        'JobRole_Sales Representative', 'MaritalStatus_Divorced', 'MaritalStatus_Married',
        'MaritalStatus_Single', 'OverTime_No', 'OverTime_Yes',
        'PerformanceRating_3', 'PerformanceRating_4',
        'RelationshipSatisfaction_1', 'RelationshipSatisfaction_2', 'RelationshipSatisfaction_3', 'RelationshipSatisfaction_4',
        'StockOptionLevel_0', 'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
        'WorkLifeBalance_1', 'WorkLifeBalance_2', 'WorkLifeBalance_3', 'WorkLifeBalance_4'
    ]

    # Asegurarse de que todas las columnas esperadas estén presentes
    for column in expected_encoded_columns:
        if column not in categorical_encoded:
            # Agregar columnas faltantes con valor 0
            categorical_encoded[column] = 0

    # Reordenar las columnas según el orden esperado
    categorical_encoded = categorical_encoded[expected_encoded_columns]

    print("Datos codificados después de agregar las columnas faltantes:")
    print(categorical_encoded)  # Imprime los datos codificados para depuración

    # Retornar las variables categóricas codificadas
    return jsonify(categorical_encoded.to_dict(orient="records")[0])


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario (en formato JSON)
    data = request.json
    print("Datos recibidos para la red neuronal:")
    print(data)  # Imprime los datos tal como los recibe el backend

    # Verificar si las características PCA están presentes
    if 'pca_features' not in data:
        return jsonify({"error": "Faltan las características PCA en los datos."}), 400

    # Obtener las características PCA directamente (ya procesadas)
    pca_features = np.array(data['pca_features']).reshape(
        1, -1)  # Asegurar que sea 2D

    # Verificar si los datos categóricos están presentes (se espera que ya estén codificados)
    if 'categorical_data' not in data:
        return jsonify({"error": "Faltan los datos categóricos codificados."}), 400

    # Recibir los datos categóricos y codificarlos (usando la función de codificación ya definida)
    categorical_data = data['categorical_data']

    # Llamar a la función de codificación para obtener las columnas necesarias
    categorical_df_encoded = encode_categorical(categorical_data)

    # Concatenar las características PCA con las categóricas codificadas
    final_input = np.concatenate(
        [pca_features, categorical_df_encoded.values], axis=1)

    # Verificar tipo y forma de final_input
    print("Dimensiones de final_input:", final_input.shape)

    # Asegurarse de que final_input es un numpy array de tipo float32
    final_input = np.array(final_input, dtype=np.float32)

    # Hacer la predicción con la red neuronal
    prediction = model.predict(final_input)

    # Convertir la predicción en un valor binario (0 o 1)
    result = (prediction > 0.5).astype(int)

    # Devolver la predicción
    return jsonify({'prediction': result[0][0]})


if __name__ == '__main__':
    app.run(debug=True)
