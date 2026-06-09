from sklearn.metrics import classification_report

def display_values(grid, X_test, y_test):
    best_model = grid.best_estimator_
    y_values = best_model.predict(X_test)



    best_model_index = grid.best_index_


    best_metrics = {
        'accuracy': grid.cv_results_["mean_test_accuracy"][best_model_index],
        'f1_score': grid.cv_results_["mean_test_f1"][best_model_index],
        'precision': grid.cv_results_["mean_test_precision"][best_model_index],
        'recall': grid.cv_results_["mean_test_recall"][best_model_index],
    }

    for k, v in best_metrics.items():
        print(f'{k}:{v}')#imprimendo dicionario
    print(classification_report(y_test, y_values, zero_division = 0)) 