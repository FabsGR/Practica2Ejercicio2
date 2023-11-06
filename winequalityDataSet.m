% Cargar tus datos desde un archivo CSV
data = readtable('winequalityDataSetCorregido.csv');

% Separar las características (X) y las etiquetas de clase (y)
X = table2array(data(:, 1:11));  % Selecciona las primeras 11 columnas como características
y = data.quality; 

% Dividir los datos en conjuntos de entrenamiento y prueba
n = size(X, 1);  % Número total de ejemplos
n_train = round(0.8 * n);  % 80% para entrenamiento
n_test = n - n_train;  % Resto para prueba

X_train = X(1:n_train, :);
y_train = y(1:n_train);

X_test = X(n_train + 1:end, :);
y_test = y(n_train + 1:end);

% Regresión Logística Multiclase
mdl = fitmnr(X_train, y_train);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión Regresión Logística Multiclase: ' num2str(accuracy * 100) '%']);

% K-Nearest Neighbors (K-NN)
k = 5;  % Ajusta el valor de k
mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión K-NN: ' num2str(accuracy * 100) '%']);

% Support Vector Machines (SVM)
% Estandarizar las características (opcional)
X_train_std = zscore(X_train);
X_test_std = zscore(X_test);
% Entrenar un modelo de Support Vector Machines (SVM) multiclase
mdl = fitcecoc(X_train_std, y_train);
% Realizar predicciones en el conjunto de prueba
Y_test_pred = predict(mdl, X_test_std);
% Calcular la precisión
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión SVM: ' num2str(accuracy * 100) '%']);

% Naive Bayes
mdl = fitcnb(X_train, y_train);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión Naive Bayes: ' num2str(accuracy * 100) '%']);