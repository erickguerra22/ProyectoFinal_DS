import React, { useState } from "react";
import axios from "axios";
import { Container, Row, Col, Card, Button, Form, Alert } from "react-bootstrap";
import 'bootstrap/dist/css/bootstrap.min.css'; // Importar el CSS de Bootstrap

function App() {
  const [inputData, setInputData] = useState({
    Age: "",
    BusinessTravel: "",
    DailyRate: "",
    Department: "",
    DistanceFromHome: "",
    Education: "",
    EducationField: "",
    Gender: "",
    JobRole: "",
    MaritalStatus: "",
    OverTime: "",
    StockOptionLevel: "",
    MonthlyIncome: "",
    MonthlyRate: "",
    NumCompaniesWorked: "",
    PercentSalaryHike: "",
    PerformanceRating: "",
    RelationshipSatisfaction: "",
    TotalWorkingYears: "",
    YearsAtCompany: "",
    YearsInCurrentRole: "",
    YearsSinceLastPromotion: "",
    YearsWithCurrManager: "",
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [pcaData, setPcaData] = useState(null); // Para almacenar los datos transformados por PCA
  const [encodedCategoricalData, setEncodedCategoricalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputData({ ...inputData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", inputData);
      setPredictionResult(response.data.prediction);
    } catch (error) {
      setError("Error al realizar la predicción.");
    }
    setLoading(false);
  };

  const handlePcaSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Enviar datos al endpoint PCA para realizar la transformación
      const response = await axios.post("http://127.0.0.1:5000/pca", inputData);
      setPcaData(response.data.transformedData); // Guardar los datos transformados por PCA
      alert("PCA realizado con éxito. Ahora puedes realizar la predicción.");
    } catch (error) {
      setError("Error al realizar la transformación PCA.");
    }
    setLoading(false);
  };




  const handleSendNumericData = async () => {
    const numericData = {
      Age: inputData.Age,
      DailyRate: inputData.DailyRate,
      DistanceFromHome: inputData.DistanceFromHome,
      HourlyRate: inputData.HourlyRate,
      MonthlyIncome: inputData.MonthlyIncome,
      NumCompaniesWorked: inputData.NumCompaniesWorked,
      PercentSalaryHike: inputData.PercentSalaryHike,
      TotalWorkingYears: inputData.TotalWorkingYears,
      TrainingTimesLastYear: inputData.TrainingTimesLastYear,
      YearsAtCompany: inputData.YearsAtCompany,
      YearsInCurrentRole: inputData.YearsInCurrentRole,
      YearsSinceLastPromotion: inputData.YearsSinceLastPromotion,
      YearsWithCurrManager: inputData.YearsWithCurrManager
    };

    console.log("Número de características:", Object.keys(numericData).length);  // Debe ser 14

    try {
      const response = await axios.post('http://127.0.0.1:5000/apply_pca', numericData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      // Actualizar el estado con las características PCA
      const pcaFeatures = response.data.pca_features;
      console.log("Características PCA recibidas:", pcaFeatures);

      if (pcaFeatures && pcaFeatures.length > 0) {
        // Aquí actualizamos el estado con los datos de PCA
        setPcaData(pcaFeatures);  // Asegúrate de que el estado 'pcaData' sea actualizado correctamente
      }
    } catch (error) {
      console.error("Error al enviar los datos:", error);
      if (error.response) {
        console.error("Respuesta del servidor:", error.response.data);
      }
    }
  };


  const handlePredictionSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pca_features: pcaData,  // Datos PCA procesados
          categorical_data: encodedCategoricalData  // Datos categóricos codificados
        }),
      });

      if (!response.ok) {
        throw new Error('Error al hacer la predicción');
      }

      const result = await response.json();
      console.log('Predicción:', result.prediction);

    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleCategoricalSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/encode_categorical', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),  // Enviar las variables categóricas
      });

      if (!response.ok) {
        throw new Error('Error al enviar las variables categóricas');
      }

      const encodedData = await response.json();

      // Imprimir los datos codificados en la consola
      console.log('Datos codificados:', encodedData);

      // Almacenar las variables categóricas codificadas en el estado
      setEncodedCategoricalData(encodedData);

    } catch (error) {
      console.error('Error:', error);
    }
  };



  return (
    <div className="App">
      <Container className="my-4">
        <h1 className="text-center mb-4" style={{ color: "#007BFF" }}>Predicción de Attrition</h1>

        <Row className="justify-content-center">
          <Col md={6}>
            <Card className="shadow-lg">
              <Card.Body>
                <Card.Title className="text-center mb-4">Ingrese los Datos</Card.Title>
                <Form onSubmit={(e) => e.preventDefault()}>
                  <h5>Variables Numéricas</h5>
                  <Row>
                    {[
                      "Age",
                      "DailyRate",
                      "DistanceFromHome",
                      "HourlyRate",
                      "MonthlyIncome",
                      "NumCompaniesWorked",
                      "PercentSalaryHike",
                      "TotalWorkingYears",
                      "TrainingTimesLastYear",
                      "YearsAtCompany",
                      "YearsInCurrentRole",
                      "YearsSinceLastPromotion",
                      "YearsWithCurrManager"
                    ].map((key) => (
                      <Col md={6} key={key}>
                        <Form.Group controlId={key}>
                          <Form.Label>{key}</Form.Label>
                          <Form.Control
                            type="number"
                            name={key}
                            value={inputData[key] || ""}
                            onChange={handleInputChange}
                            required
                          />
                        </Form.Group>
                      </Col>
                    ))}
                  </Row>
                  <Button variant="primary" type="button" onClick={handleSendNumericData}>
                    Enviar Datos Numéricos
                  </Button>
                </Form>

                <hr />

                {/* Variables categóricas */}
                <h5>Variables Categóricas</h5>
                <Row>
                  <Col md={6}>
                    <Form.Group controlId="BusinessTravel">
                      <Form.Label>Business Travel</Form.Label>
                      <Form.Control
                        as="select"
                        name="BusinessTravel"
                        value={inputData.BusinessTravel}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Travel_Rarely">Travel Rarely</option>
                        <option value="Travel_Frequently">Travel Frequently</option>
                        <option value="Non-Travel">Non-Travel</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  {/* Resto de las variables categóricas */}
                  <Col md={6}>
                    <Form.Group controlId="Department">
                      <Form.Label>Department</Form.Label>
                      <Form.Control
                        as="select"
                        name="Department"
                        value={inputData.Department}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Sales">Sales</option>
                        <option value="Research & Development">Research & Development</option>
                        <option value="Human Resources">Human Resources</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="Education">
                      <Form.Label>Education</Form.Label>
                      <Form.Control
                        as="select"
                        name="Education"
                        value={inputData.Education}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="1">Below College</option>
                        <option value="2">College</option>
                        <option value="3">Bachelor</option>
                        <option value="4">Master</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="EducationField">
                      <Form.Label>Education Field</Form.Label>
                      <Form.Control
                        as="select"
                        name="EducationField"
                        value={inputData.EducationField}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="1">Other</option>
                        <option value="2">Life Sciences</option>
                        <option value="3">Medical</option>
                        <option value="4">Marketing</option>
                        <option value="5">Technical Degree</option>
                        <option value="6">Human Resources</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="Gender">
                      <Form.Label>Gender</Form.Label>
                      <Form.Control
                        as="select"
                        name="Gender"
                        value={inputData.Gender}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="JobRole">
                      <Form.Label>Job Role</Form.Label>
                      <Form.Control
                        as="select"
                        name="JobRole"
                        value={inputData.JobRole}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Sales Executive">Sales Executive</option>
                        <option value="Research Scientist">Research Scientist</option>
                        <option value="Laboratory Technician">Laboratory Technician</option>
                        <option value="Manufacturing Director">Manufacturing Director</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="MaritalStatus">
                      <Form.Label>Marital Status</Form.Label>
                      <Form.Control
                        as="select"
                        name="MaritalStatus"
                        value={inputData.MaritalStatus}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Single">Single</option>
                        <option value="Married">Married</option>
                        <option value="Divorced">Divorced</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group controlId="OverTime">
                      <Form.Label>OverTime</Form.Label>
                      <Form.Control
                        as="select"
                        name="OverTime"
                        value={inputData.OverTime}
                        onChange={handleInputChange}
                        required
                      >
                        <option value="">Seleccionar</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </Form.Control>
                    </Form.Group>
                  </Col>

                  <Button variant="primary" type="button" onClick={handleCategoricalSubmit}>
                    Codificar Variables Categóricas
                  </Button>
                </Row>

                <Button variant="primary" type="button" onClick={handlePredictionSubmit} disabled={!pcaData}>
                  Predecir con PCA
                </Button>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        <Row className="justify-content-center mt-4">
          <Col md={6}>
            {error && <Alert variant="danger">{error}</Alert>}
            {predictionResult !== null && (
              <Card className="shadow-lg">
                <Card.Body>
                  <Card.Title className="text-center">Resultado de la Predicción</Card.Title>
                  <p className="text-center" style={{ fontSize: "1.2em" }}>
                    <strong>Predicción de Attrition:</strong> {predictionResult === 0 ? "No" : "Sí"}
                  </p>
                </Card.Body>
              </Card>
            )}
          </Col>
        </Row>
      </Container>
    </div>
  );

}
export default App;
