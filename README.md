## 🚀 Model Deployment
### Local
To launch a frontend that classifies images with the trained model, use the ```make launch-local``` command.

### Production
In order to deploy a system like this into production, the most important considerations would be:
- **Deliverable**:
    - Define the deliverable format (API endpoint, frontend for user interaction...) and implement accordingly
- **Infrastructure**: 
    - Use a cloud provider for scalable computation (AWS/Azure/GCP)
    - Use kubernetes or similar service to orchestrate a scalable GPU-based cluster
- **Model Serving**:
    - Optimize inference time by optimizing the model and using serving platforms such as NVIDIA Triton
- **CI/CD**:
    - Automate deployments by using Git actions or Jenkins to build the Docker image
    - Use cloud provider image registry to deploy new images
- **Storage/Logging**:
    - Use cloud storage to store large artifacts (AWS S3, Google Cloud Storage...) such as the generated images and the models
    - Ensure a good logging and monitoring of service usage
- **Security**:
    - Protect endpoints/frontend with access management policies

### Next Steps
- [] Different training strategies such as freezing model backbone during training
- [] Experiment different data splits, to ensure a robust evaluation, including cross validation methods
- [] Experiment with different architectures
- [] Activation maps to further understand where model may be focusing
- [] Understand production requirements (latency, hardware...) and adjust strategy accordingly
