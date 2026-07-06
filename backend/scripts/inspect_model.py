import joblib
try:
    model = joblib.load('router_model.pkl')
    print(f"Model Class: {type(model)}")
    if hasattr(model, 'steps'):
        print("Pipeline Steps:")
        for name, step in model.steps:
            print(f" - {name}: {type(step)}")
            if name == 'clf' and hasattr(step, 'estimator'):
                print(f"   -> Estimator: {type(step.estimator)}")
    
    # If it's a multi-label wrapper
    if hasattr(model, 'estimator'):
         print(f"Base Estimator: {type(model.estimator)}")

except Exception as e:
    print(f"Error loading model: {e}")
