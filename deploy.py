
import boto3
from botocore.exceptions import ClientError
import time

REGION = "ap-southeast-2"

ROLE_ARN = (
    "arn:aws:iam::071732232580:role/"
    "SageMakerIrisExecutionRole"
)

MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:ap-southeast-2:071732232580:"
    "model-package/IrisClassifier/1"
)

MODEL_NAME = "iris-classifier-model"
CONFIG_NAME = "iris-classifier-config"
ENDPOINT_NAME = "iris-classifier-endpoint"

INSTANCE_TYPE = "ml.t2.medium"

sm = boto3.client("sagemaker", region_name=REGION)


def resource_exists(operation, **kwargs):
    try:
        getattr(sm, operation)(**kwargs)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def deploy_model():
    if not resource_exists("describe_model", ModelName=MODEL_NAME):
        sm.create_model(
            ModelName=MODEL_NAME,
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer={
                "ModelPackageName": MODEL_PACKAGE_ARN
            }
        )
        print("Model created successfully.")
    else:
        print("Model already exists.")

    if not resource_exists(
        "describe_endpoint_config",
        EndpointConfigName=CONFIG_NAME
    ):
        sm.create_endpoint_config(
            EndpointConfigName=CONFIG_NAME,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": MODEL_NAME,
                    "InitialInstanceCount": 1,
                    "InstanceType": INSTANCE_TYPE,
                    "InitialVariantWeight": 1.0
                }
            ]
        )
        print("Endpoint configuration created.")
    else:
        print("Endpoint configuration already exists.")

    if resource_exists(
        "describe_endpoint",
        EndpointName=ENDPOINT_NAME
    ):
        print("Endpoint already exists. No new endpoint created.")
        return

    print("Creating endpoint. Hosting charges may apply.")

    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=CONFIG_NAME
    )

    while True:
        response = sm.describe_endpoint(
            EndpointName=ENDPOINT_NAME
        )

        status = response["EndpointStatus"]
        print("Endpoint status:", status)

        if status == "InService":
            print("Deployment successful!")
            break

        if status in ("Failed", "OutOfService"):
            raise RuntimeError(
                response.get("FailureReason", status)
            )

        time.sleep(30)


if __name__ == "__main__":
    deploy_model()
