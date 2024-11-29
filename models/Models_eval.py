
from sklearn.model_selection import GridSearchCV

def grid_search_cv_eval(X,Y, models, param_grid, scorings,cross_validation):
    """Here we evaluate the models using gridsearchcv and returns the dictionary of best models and result."""

    best_models = {}
    result = {}
    for model in models:
        print(result)
        print(f"\nRunning GridSearch for {model}...")
        gsv = GridSearchCV(
            estimator=models[model],
            param_grid=param_grid[model],
            cv=cross_validation,
            scoring=scorings,
            refit='accuracy'  # Primary metric for model selection
        )
        gsv.fit(X, Y)
        best_models[model] = gsv.best_estimator_
        best_index = gsv.best_index_
        print(f'Best parameters for {model}: {gsv.best_params_}')
        print(f'Best accuracy: {gsv.cv_results_["mean_test_accuracy"][best_index]:.4f}')
        print(f'Best precision: {gsv.cv_results_["mean_test_precision"][best_index]:.4f}')
        print(f'Best recall: {gsv.cv_results_["mean_test_recall"][best_index]:.4f}')
        result[model] = {"parameter":gsv.best_params_,"accuracy":gsv.cv_results_["mean_test_accuracy"][best_index], "precision": gsv.cv_results_["mean_test_precision"][best_index],"recall": gsv.cv_results_["mean_test_recall"][best_index]}

        return best_models, result