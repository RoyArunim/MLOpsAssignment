Heart Disease Risk Prediction - End-to-End MLOps Pipeline

This repository contains a complete, production-ready Machine Learning pipeline for predicting heart disease risk based on the UCI Heart Disease dataset. It demonstrates MLOps best practices including model tracking, containerized model serving via FastAPI, Kubernetes orchestration, and real-time observability using Prometheus.

🏗️ Architecture & Tech Stack

Modeling: scikit-learn (Random Forest Classifier, scikit-learn Pipelines)

API Framework: FastAPI, Pydantic, Uvicorn

Containerization: Docker

Orchestration: Kubernetes (Minikube local cluster)

Observability: Custom ASGI Logging Middleware, Prometheus Metrics

CI/CD: GitHub Actions (Pytest, Flake8)

⚙️ Prerequisites

Ensure you have the following installed on your local machine or VM:

Python 3.10+

Docker

Minikube & kubectl

Git

🚀 Step 1: Local Setup & Model Training

Clone the repository:

git clone <YOUR_REPO_URL>
cd my-ml-project


Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Train the model and generate the artifact:
(Ensure your EDA/training script is run to generate the model.pkl file before building the container)

python train.py 


🐳 Step 2: Docker Containerization

Build the production Docker image. We use a specific version tag (v3) to avoid Kubernetes aggressive caching issues with :latest.

docker build -f DOCKERFILE -t heart-disease-api:v3 .

Install minikube and kubectl
# 1. Download and install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# 2. Download and install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

☸️ Step 3: Kubernetes Deployment (Minikube)

Start the Minikube cluster:

minikube start


Load the local Docker image into the Minikube registry:

minikube image load heart-disease-api:v3


Deploy the application and LoadBalancer service:

kubectl apply -f k8s-deployment.yaml


Verify the pods are running:

kubectl get pods


Expose the LoadBalancer (Crucial for Minikube):
Open a new, separate terminal window and run the following command. Leave it running in the background:

minikube tunnel


Retrieve your External IP address:

kubectl get svc heart-disease-service


Note the EXTERNAL-IP for the next steps.

🧪 Step 4: Testing the API

Send a test payload to the prediction endpoint using your EXTERNAL-IP.

curl -X POST "http://<YOUR_EXTERNAL_IP>:80/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 63,
           "sex": 1,
           "cp": 3,
           "trestbps": 145,
           "chol": 233,
           "fbs": 1,
           "restecg": 0,
           "thalach": 150,
           "exang": 0,
           "oldpeak": 2.3,
           "slope": 0,
           "ca": 0,
           "thal": 1
         }'


Expected Output:

{"prediction": 1, "confidence": 0.8643, "risk_status": "High Risk"}


📊 Step 5: Observability & Monitoring

1. View Custom API Logs

The API features structured middleware logging. To view the request logs:

kubectl logs deployment/heart-disease-deployment


2. View Raw Prometheus Metrics

The API automatically instruments RED metrics (Rate, Errors, Duration).

curl http://<YOUR_EXTERNAL_IP>:80/metrics


3. Launch the Prometheus Dashboard

To visualize the API traffic, spin up a lightweight Prometheus dashboard:

# Create the config file
cat << 'EOF' > prometheus.yml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'heart-disease-api'
    static_configs:
      - targets: ['<YOUR_EXTERNAL_IP>:80']
EOF

# Run the Prometheus container
docker run -d --name prometheus-dashboard --network host -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus


Access the dashboard by navigating to http://localhost:9090 in your web browser. Type http_requests_total in the search bar and click the Graph tab to view your API traffic visually.

🧹 Step 6: Cleanup

To safely tear down the environment and reclaim resources:

# Stop the Minikube tunnel (Ctrl+C in its terminal)
minikube stop
minikube delete
docker stop prometheus-dashboard
docker rm prometheus-dashboard
