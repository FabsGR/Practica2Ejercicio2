% Cargar tus datos desde un archivo CSV
data = readtable('AutoInsurSwedenDataset.csv');  % Ajusta el nombre del archivo

% Obtener las características (X) y las etiquetas de clase (Y)
X = data.X;
Y = data.Y;

% Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento)
n = length(X);  % Número total de ejemplos
n_train = round(0.8 * n);  % 80% para entrenamiento
n_test = n - n_train;  % Resto para prueba

X_train = X(1:n_train);
Y_train = Y(1:n_train);

X_test = X(n_train + 1:end);
Y_test = Y(n_train + 1:end);

% A continuación, puedes entrenar diferentes modelos de clasificación y evaluar su rendimiento.

% Regresión Logística
mdl_logistica = fitmnr(X_train, Y_train);
Y_test_pred_logistica = predict(mdl_logistica, X_test);
precision_logistica = sum(Y_test_pred_logistica == Y_test) / n_test;

% K-Nearest Neighbors (K-NN)
k = 5;  % Ajusta el valor de k según tus necesidades
mdl_knn = fitcknn(X_train, Y_train, 'NumNeighbors', k);
Y_test_pred_knn = predict(mdl_knn, X_test);
precision_knn = sum(Y_test_pred_knn == Y_test) / n_test;

% Support Vector Machines (SVM)
mdl_svm = fitcecoc(X_train, Y_train);
Y_test_pred_svm = predict(mdl_svm, X_test);
precision_svm = sum(Y_test_pred_svm == Y_test) / n_test;

% Naive Bayes
mdl_naive_bayes = fitcecoc(X_train, Y_train);
Y_test_pred_naive_bayes = predict(mdl_naive_bayes, X_test);
precision_naive_bayes = sum(Y_test_pred_naive_bayes == Y_test) / n_test;

% Mostrar resultados de precisión
disp(['Precisión Regresión Logística: ' num2str(precision_logistica * 100) '%']);
disp(['Precisión K-NN: ' num2str(precision_knn * 100) '%']);
disp(['Precisión SVM: ' num2str(precision_svm * 100) '%']);
disp(['Precisión Naive Bayes: ' num2str(precision_naive_bayes * 100) '%']);
