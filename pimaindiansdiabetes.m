% Cargar tus datos desde un archivo CSV o cualquier fuente de datos
data = readtable('pimaindiansdiabetesDataset.csv');  % Ajusta el nombre del archivo

% Separar las características (X) y las etiquetas de clase (y)
X = table2array(data(:, 1:8));  % Selecciona las primeras 8 columnas como características
y = data.Var9;  % Ajusta el nombre de la columna que contiene las etiquetas de clase

% Obtener el número total de ejemplos
n = size(X, 1);

% Determinar el porcentaje de datos para entrenamiento (80%)
train_percentage = 0.8;

% Generar una permutación aleatoria de los índices
rng('default');  % Establecer la semilla aleatoria para reproducibilidad
random_indices = randperm(n);

% Calcular el número de ejemplos para entrenamiento
n_train = round(train_percentage * n);
n_test = n - n_train;  % Resto para prueba

% Obtener los índices para los datos de entrenamiento y prueba
train_indices = random_indices(1:n_train);
test_indices = random_indices(n_train + 1:end);

% Dividir los datos en conjuntos de entrenamiento y prueba
X_train = X(train_indices, :);
y_train = y(train_indices);

X_test = X(test_indices, :);
y_test = y(test_indices);

% Regresión Logística Multiclase
mdl = fitmnr(X_train, y_train);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión Regresión Logística Multiclase: ' num2str(accuracy * 100) '%']);

% K-Nearest Neighbors (K-NN)
k = 5;  % Ajusta el valor de k según tus necesidades
mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión K-NN: ' num2str(accuracy * 100) '%']);

% Support Vector Machines (SVM)
mdl = fitcsvm(X_train, y_train);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión SVM: ' num2str(accuracy * 100) '%']);

% Naive Bayes
mdl = fitcnb(X_train, y_train);
Y_test_pred = predict(mdl, X_test);
accuracy = sum(Y_test_pred == y_test) / n_test;
disp(['Precisión Naive Bayes: ' num2str(accuracy * 100) '%']);
