
from sklearn.model_selection import GridSearchCV

def grid_search_cv_eval(X,Y, models, param_grid, scorings,cross_validation):
    """Here we evaluate the models using gridsearchcv and returns the dictionary of best models and result."""

    best_models = {}
    result = {}
    print(models)
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


from sklearn.model_selection import cross_val_score

# Define the function to compare models with default parameters
def evaluate_models(X, Y, models, scorings, cross_validation):
    result = {}
    
    # Loop through models and evaluate each one
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Evaluate using the defined scoring metrics
        model_scores = {}
        for score_name, scorer in scorings.items():
            score = cross_val_score(model, X, Y, cv=cross_validation, scoring=scorer)
            model_scores[score_name] = score.mean()
        
        # Store results
        result[model_name] = model_scores
        
        # Print the results
        print(f"Accuracy: {model_scores['accuracy']:.4f}")
        print(f"Precision: {model_scores['precision']:.4f}")
        print(f"Recall: {model_scores['recall']:.4f}")
        print(f"F1 Score: {model_scores['f1']:.4f}")
    
    return result