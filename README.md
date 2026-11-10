\# Iris Flower Classification and Deployment Using AWS SageMaker



\## 1. Introduction



This project demonstrates how to train a Machine Learning model and deploy it using Amazon Web Services (AWS) SageMaker.



The Iris dataset is used to classify flowers into three categories:

\- Setosa

\- Versicolor

\- Virginica



A Random Forest Classifier is trained and deployed as a managed SageMaker endpoint.



\## 2. Technologies Used



\- Python 3.12

\- Scikit-learn

\- Pandas and NumPy

\- Joblib

\- Boto3

\- Amazon SageMaker

\- Amazon S3

\- AWS Model Registry



\## 3. Model Training



The Iris dataset is divided into training and testing sets using an 80:20 split.



A Random Forest Classifier is trained with 100 decision trees.



The model achieved the following accuracy:



\*\*Model Accuracy: 90% (0.9000)\*\*



The trained model is saved as `model.joblib`.



\## 4. AWS SageMaker Deployment



The trained model was uploaded to an Amazon S3 bucket.



The model was registered in SageMaker Model Registry under the group `IrisClassifier`.



A SageMaker model and endpoint configuration were created. The model was deployed using a real-time endpoint with the `ml.t2.medium` instance type in the Sydney region (`ap-southeast-2`).



The endpoint successfully reached the `InService` state.



\## 5. Endpoint Testing



The following sample input was sent to the deployed endpoint:



```json

{"instances": \[\[5.1, 3.5, 1.4, 0.2]]}

```



The endpoint returned:



```json

{"predictions": \["setosa"]}

```



This confirms that the deployed model successfully generated a prediction.



\## 6. Benefits of Managed Cloud Machine Learning Services



Managed cloud services such as AWS SageMaker provide infrastructure for training, deploying and monitoring machine learning models.



Benefits include:

\- Reduced infrastructure management.

\- Easy model deployment.

\- Scalable computing resources.

\- Integration with cloud storage.

\- Faster development and experimentation.



\## 7. Why Should an Endpoint Be Deleted After Testing?



A real-time endpoint uses cloud computing resources while it is running.



AWS may charge for these resources even when the endpoint is not receiving prediction requests.



Deleting an unused endpoint helps prevent unnecessary costs.



The trained model and registered model package can be retained separately.



\## 8. Difference Between Model Registry and Endpoint



\*\*Model Registry:\*\* Stores model versions, metadata and approval status. It helps manage and track machine learning models.



\*\*Endpoint:\*\* Hosts a deployed model and receives input data to generate predictions.



A registered model is not automatically deployed. It must be deployed separately to an endpoint.



\## 9. Managed ML Service vs Kubernetes Deployment



A managed machine learning service such as SageMaker provides built-in tools for model deployment and hosting. The cloud provider manages much of the underlying infrastructure.



Kubernetes is a container orchestration platform. It provides more control and flexibility, but users generally need to manage more infrastructure, configuration, scaling and monitoring.



Managed services are often easier for beginners, while Kubernetes is useful when greater infrastructure control is required.



\## 10. Conclusion



This project successfully demonstrated the complete machine learning deployment workflow using AWS SageMaker.



The Iris classification model achieved 90% accuracy and was successfully deployed to a managed endpoint.



The endpoint returned the expected prediction, `setosa`, for the provided sample.



After testing, the endpoint was deleted to avoid ongoing hosting charges.



\## 11. Project Files



\- `train.py` – Model training.

\- `deploy.py` – SageMaker deployment.

\- `test\_endpoint.py` – Endpoint testing.

\- `inference.py` – Model inference functions.

\- `model.joblib` – Trained model.

\- `model.tar.gz` – Model archive for SageMaker.

\- `requirements.txt` – Python dependencies.

\- `screenshots/` – Assignment screenshots.



