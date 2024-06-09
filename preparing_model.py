def prepare_model():
    import os
    import pickle
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import optuna

    # prepare data
    input_dir = "clf-data"
    categories = ["empty", "not_empty"]

    data = []
    labels = []
    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(input_dir, category)):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    # train classifiers

    # Load dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    def objective(trial):
        # Suggest hyperparameters
        gamma = trial.suggest_uniform("gamma", 0.0001, 0.01)
        C = trial.suggest_uniform("C", 1, 1000)

        # Train and evaluate model
        model = SVC(gamma=gamma, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred, y_test)
        return score

    # create a study object
    study = optuna.create_study(direction="maximize")
    # optimize object function
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params

    print("Best hyperparameters:", study.best_params)
    print(f"{str(study.best_value * 100)}% of samples were correctly classified")

    # using optimal hyperparameters
    best_estimator = SVC(**best_params)
    best_estimator.fit(X_train, y_train)
    y_prediction = best_estimator.predict(X_test)

    score = accuracy_score(y_prediction, y_test)

    # save model
    pickle.dump(best_estimator, open("./model.p", "wb"))
