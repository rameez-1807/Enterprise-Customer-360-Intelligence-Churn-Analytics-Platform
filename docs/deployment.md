# 🚀 Enterprise Cloud Deployment Guide

This document outlines the procedures for deploying the **Enterprise Customer 360 Intelligence & Churn Analytics Platform** to production environments.

---

## 💻 Environment Variables Configuration

Before deploying, ensure you configure the following environment variables in your runtime settings:

| Variable Name | Required | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | Yes | `PROD` | Runtime profile environment (`DEV`, `STAGING`, `PROD`). |
| `LOG_LEVEL` | No | `INFO` | Output verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `RAW_DATA_DIR` | No | `data/raw` | Local directory for raw upstream datasets. |
| `MODEL_ARTIFACTS_DIR` | No | `data/outputs/models` | Storage path for serialized LightGBM/SHAP artifacts. |

---

## 🐳 Containerized Deployments (Recommended)

### 1. Docker Build
Use the multi-stage `Dockerfile` to create a production image:
```bash
docker build -t customer-360-platform:latest .
```

### 2. Docker Compose Local Run
Run the full environment locally with mounted volumes:
```bash
docker-compose up -d
```

---

## ☁️ Cloud Providers Integration

### A. AWS Elastic Container Service (ECS)
1. **Container Registry**: Push the Docker image to AWS ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
   docker tag customer-360-platform:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/customer-360-platform:latest
   docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/customer-360-platform:latest
   ```
2. **Task Definition**: Create an ECS Fargate Task Definition exposing container port `8501`.
3. **Application Load Balancer (ALB)**: Route traffic from port `80`/`443` to the task group on port `8501`. Enable sticky sessions if scale is $>1$ task instance.

### B. Azure App Services (Web App for Containers)
1. Push your docker image to **Azure Container Registry (ACR)**.
2. Create an **App Service** using the image.
3. Configure application settings in Azure portal:
   * `WEBSITES_PORT = 8501` (Informs Azure App Service which port to forward).
4. Map your custom domain and enable HTTPS.

### C. Google Cloud Run
Deploy containerized workloads directly with automatic scaling:
```bash
gcloud run deploy customer-360-platform \
  --image gcr.io/<gcp_project_id>/customer-360-platform:latest \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated
```

---

## 🚀 PaaS & Serverless Platforms

### D. Streamlit Community Cloud
1. Fork this repository to your GitHub profile.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New App**, select your forked repository, the `master/main` branch, and set the file path to `src/dashboard/app.py`.
4. Click **Deploy**. (The app automatically installs requirements and starts).

### E. Railway & Render
These platforms use the native `Dockerfile` or `requirements.txt` to compile.
* **Railway**: Connect your GitHub repository. Railway will detect the `Dockerfile` and deploy the dashboard. Expose port `8501` under the variables panel.
* **Render**: Create a new **Web Service**, connect your repository, select **Docker** as the environment, and specify the port `8501` in Settings.
