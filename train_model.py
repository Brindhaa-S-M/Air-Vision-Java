from backend.ml import train_model

if __name__ == "__main__":
    metrics = train_model()
    print("AirVision Random Forest model trained.")
    print("Evaluation metrics (synthetic demonstration data only):")
    for key, value in metrics.items():
        print(f"{key}: {value}")
