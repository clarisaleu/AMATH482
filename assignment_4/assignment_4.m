%{ 
    Assignment #4 - Classifying Digits (MNIST)
    AMATH482 - Computational Methods For Data Science - Mar. 10th, 2021
    Taught by Professor Jason J. Bramburger (Winter 2021)
    Written By: Clarisa Leu-Rodriguez - email: cleu@uw.edu
%}


%% Import & Parse MNIST Data
clear all; close all; clc; 
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte1', ...
    'train-labels-idx1-ubyte1');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte1', ...
    't10k-labels-idx1-ubyte1');


%% Part #1 - Perform SVD Analysis of Digit Images & Low-Rank Approximation
% Convert image data to double to do calculations.
images_train = im2double(images_train);
images_test = im2double(images_test);

% Resize Data - Each column vector is a different image.
S_train = size(images_train);
S_test = size(images_test);
dat_train = reshape(images_train, [S_train(1) * S_train(2), S_train(3)]);
dat_test = reshape(images_test, [S_test(1) * S_test(2), S_test(3)]);

% Plot first 8 images of training data and test data.
figure();
image_dim = 28;  % Images are 28 * 28 = 784 pixels.
for k = 1:8
    subplot(4, 4, k);
    im_check = reshape(im2uint8(dat_train(:, k)), image_dim, image_dim);
    imshow(im_check);
    title(strcat("Training Data - Image #", num2str(k)));
    set(gca, 'fontsize', 13);
end

% Plot first 8 images of test data.
for k = 1:8
    subplot(4, 4, k + 8);
    im_check = reshape(im2uint8(dat_test(:, k)), image_dim, image_dim);
    imshow(im_check);
    title(strcat("Test Data - Image #", num2str(k)));
    set(gca, 'fontsize', 13);
end

dat_train = dat_train';
dat_test = dat_test';

% Get training data row wise mean to scale test & train data.
[m, n] = size(dat_train);
[m1, n2] = size(dat_test);
mn = mean(dat_train, 1);
sub1 = repmat(mn, m, 1);
sub2 = repmat(mn, m1, 1);

% Subtract the mean pixel location from each row.
dat_train = dat_train - sub1;
dat_test = dat_test - sub2;
dat_train = dat_train';
dat_test = dat_test';

% Perform SVD with econ mode.
[U, S, V] = svd (dat_train, 'econ');
lambdas = diag(S).^2;  % Look at singular values.
energies = lambdas ./ sum(lambdas);  % Save energies to plot later.

% Find rank r of the digital space such that 90% energy is met.
% To meet 90% energy - rank of 87 needed which accounts for 90.01% energy.
energy = 0;
total_energy = sum(diag(S).^2);
threshold = 0.9; 
r = 0;
while energy <= threshold
    r = r + 1;
    energy = energy + S(r,r)^2 / total_energy;
end

% Plot Energies and Singular Values
figure();
subplot(2, 1, 1);
plot(energies, 'k.', 'markersize', 20);
title('Energies'); xlabel('Mode'); 
ylabel('Percentage of Energy Captured in Mode');
grid on; set(gca, 'Fontsize', 13);

subplot(2, 1, 2);
plot(lambdas, 'k.', 'markersize', 20);
title('Singular Values'); xlabel('Mode'); 
ylabel('Singular Value of Mode');
grid on; set(gca, 'Fontsize', 13);


% Look at image reconstruction of low-rank approximation of first 8 images
% in training data and plot first 8 principal components.
figure();
x_train_approx = U(:, 1:r) * S(1:r, 1:r) * (V(:, 1:r)');
for k = 1:8
    subplot(4, 4, k);
    im_check = reshape(im2uint8(x_train_approx(:, k)), image_dim, image_dim);
    imshow(im_check);
    title(strcat("Low Rank Approx. - Image #", num2str(k)));
    set(gca, 'fontsize', 13);
end

% Plot first 8 principal components.
for k = 1:8
    subplot(4, 4, k + 8);
    ut1 = reshape(U(:, k), image_dim, image_dim);
    ut2 = rescale(ut1);
    imshow(ut2);
    title(strcat("Principal Component #", num2str(k)));
    set(gca, 'fontsize', 13);
end

% Plot 3D - Project onto first three V-modes (columns) colored by their 
% digit label.
xs = U(:, 1)' * dat_train;
ys = U(:, 2)' * dat_train;
zs = U(:, 3)' * dat_train;

zeros_indices = find(labels_train == 0);
ones_indices = find(labels_train == 1);
twos_indices = find(labels_train == 2);
threes_indices = find(labels_train == 3);
fours_indices = find(labels_train == 4);
fives_indices = find(labels_train == 5);
sixes_indices = find(labels_train == 6);
sevens_indices = find(labels_train == 7);
eights_indices = find(labels_train == 8);
nines_indices = find(labels_train == 9);

figure();
plot3(xs(zeros_indices), ys(zeros_indices), zs(zeros_indices), 'o', ...
    xs(ones_indices), ys(ones_indices), zs(ones_indices), 'o', ...
    xs(twos_indices), ys(twos_indices), zs(twos_indices), 'o', ...
    xs(threes_indices), ys(threes_indices), zs(threes_indices), 'o', ...
    xs(fours_indices), ys(fours_indices), zs(fours_indices), 'o', ...
    xs(fives_indices), ys(fives_indices), zs(fives_indices), 'o', ...
    xs(sixes_indices), ys(sixes_indices), zs(sixes_indices), 'o', ...
    xs(sevens_indices), ys(sevens_indices), zs(sevens_indices), 'o', ...
    xs(eights_indices), ys(eights_indices), zs(eights_indices), 'o', ...
    xs(nines_indices), ys(nines_indices), zs(nines_indices), 'o');

title('Projection of Training Data onto First Three Modes');
xlabel('Mode 1'); ylabel('Mode 2'); zlabel('Mode 3');
legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
grid on; set(gca, 'Fontsize', 13);


%% Part 2: Build Classifiers to Identify Individual Digits in Training Set

% Get data into PCA space - 90% energy captured
project_dat = U(:, 1:r)' * dat_train;
project_dat_test = U(:, 1:r)' * dat_test;

% Build LDA, SVM, and decision tree for every pair of two digits
percent_correct_lda_two_digits_test = [];
percent_correct_svm_two_digits_test = [];
percent_correct_decis_tree_two_digits_test = [];
percent_correct_lda_two_digits_train = [];
percent_correct_svm_two_digits_train = [];
percent_correct_decis_tree_two_digits_train = [];
for i = 0:9
    % For digit i - get the training/test data and their respective labels.
    test_indices_i = find(labels_test == i);
    train_indices_i = find(labels_train == i);
    train_dat_i = project_dat(:, train_indices_i);
    test_dat_i = project_dat_test(:, test_indices_i);
    train_labels_i = i.* ones(length(train_dat_i), 1);
    test_labels_i = i.* ones(length(test_dat_i), 1);
    
    for j = i+1:9
        % For digit j - get the training/test data and their respective
        % labels.
        test_indices_j = find(labels_test == j);
        train_indices_j = find(labels_train == j);
        train_dat_j = project_dat(:, train_indices_j);
        test_dat_j = project_dat_test(:, test_indices_j);    
        train_labels_j = j.* ones(length(train_dat_j), 1);
        test_labels_j = j.* ones(length(test_dat_j), 1);
        
        % Combine training/test data and labels for digit i and j.
        train_dat_combined = [train_dat_i, train_dat_j];
        test_dat_combined = [test_dat_i, test_dat_j];
        train_labels_combined = [train_labels_i; train_labels_j];
        test_labels_combined = [test_labels_i; test_labels_j];
        
        % Perform LDA
        class_test = classify(test_dat_combined', train_dat_combined', ...
            train_labels_combined, 'linear');
        class_train = classify(train_dat_combined', train_dat_combined', ...
            train_labels_combined, 'linear');
        equal_test = class_test == test_labels_combined;
        equal_train = class_train == train_labels_combined;
        number_right_test = sum(equal_test(:) == 1);
        number_right_train = sum(equal_train(:) == 1);
        percent_right_test = number_right_test / (length(test_labels_combined));
        percent_right_train = number_right_train / (length(train_labels_combined));
        percent_correct_lda_two_digits_test = [percent_correct_lda_two_digits_test; ...
            [i, j, percent_right_test]];
        percent_correct_lda_two_digits_train = [percent_correct_lda_two_digits_train; ...
            [i, j, percent_right_train]];

        % Do SVM
        train_dat_combined_trans = train_dat_combined';
        mdl = fitcecoc((1 / max(train_dat_combined_trans(:))).* ...
            train_dat_combined', train_labels_combined);
        test_labels = predict(mdl, (1 / ...
            max(train_dat_combined_trans(:))).* test_dat_combined');
        train_labels = predict(mdl, (1 / ...
            max(train_dat_combined_trans(:))).* train_dat_combined');
        equal_svm_test = test_labels_combined == test_labels;
        equal_svm_train = train_labels_combined == train_labels;
        number_right_svm_test = sum(equal_svm_test(:) == 1);
        number_right_svm_train = sum(equal_svm_train(:) == 1);
        percent_right_svm_test = number_right_svm_test / (length(test_labels_combined));
        percent_right_svm_train = number_right_svm_train / (length(train_labels_combined));
        percent_correct_svm_two_digits_test = [percent_correct_svm_two_digits_test; ...
            [i, j, percent_right_svm_test]];
        percent_correct_svm_two_digits_train = [percent_correct_svm_two_digits_train; ...
            [i, j, percent_right_svm_train]];

        % Decision Tree Learning
        tree = fitctree(train_dat_combined', train_labels_combined);
        test_labels_tree = predict(tree, test_dat_combined');
        train_labels_tree = predict(tree, train_dat_combined');
        equal_tree_test = test_labels_tree == test_labels_combined;
        equal_tree_train = train_labels_tree == train_labels_combined;
        number_right_tree_test = sum(equal_tree_test(:) == 1);
        number_right_tree_train = sum(equal_tree_train(:) == 1);
        percent_right_tree_test = number_right_tree_test / (length(test_labels_combined));
        percent_right_tree_train = number_right_tree_train / (length(train_labels_combined));
        percent_correct_decis_tree_two_digits_test = [percent_correct_decis_tree_two_digits_test; ...
            [i, j, percent_right_tree_test]];
        percent_correct_decis_tree_two_digits_train = [percent_correct_decis_tree_two_digits_train; ...
            [i, j, percent_right_tree_train]];
    end
end

% Build LDA, SVM, and decision tree for every triple of three digits
percent_correct_lda_three_digits_test = [];
percent_correct_svm_three_digits_test = [];
percent_correct_decis_tree_three_digits_test = [];
percent_correct_lda_three_digits_train = [];
percent_correct_svm_three_digits_train = [];
percent_correct_decis_tree_three_digits_train = [];
for i = 0:9
    % For digit i - get the training/test data and their respective labels.
    test_indices_i = find(labels_test == i);
    train_indices_i = find(labels_train == i);
    train_dat_i = project_dat(:, train_indices_i);
    test_dat_i = project_dat_test(:, test_indices_i);
    train_labels_i = i.* ones(length(train_dat_i), 1);
    test_labels_i = i.* ones(length(test_dat_i), 1);
    
    for j = i+1:9
        % For digit j - get the training/test data and their respective
        % labels.
        test_indices_j = find(labels_test == j);
        train_indices_j = find(labels_train == j);
        train_dat_j = project_dat(:, train_indices_j);
        test_dat_j = project_dat_test(:, test_indices_j);    
        train_labels_j = j.* ones(length(train_dat_j), 1);
        test_labels_j = j.* ones(length(test_dat_j), 1);
        
        for k = j+1:9
            % For digit k - get the training/test data and their respective
            % labels.
            test_indices_k = find(labels_test == k);
            train_indices_k = find(labels_train == k);
            train_dat_k = project_dat(:, train_indices_k);
            test_dat_k = project_dat_test(:, test_indices_k);    
            train_labels_k = k.* ones(length(train_dat_k), 1);
            test_labels_k = k.* ones(length(test_dat_k), 1);

            % Combine training/test data and labels for digit i, j, and k.
            train_dat_combined = [train_dat_i, train_dat_j, train_dat_k];
            test_dat_combined = [test_dat_i, test_dat_j, test_dat_k];
            train_labels_combined = [train_labels_i; train_labels_j; ...
                train_labels_k];
            test_labels_combined = [test_labels_i; test_labels_j; ...
                test_labels_k];

            % Perform LDA
            class_test = classify(test_dat_combined', train_dat_combined', ...
                train_labels_combined, 'linear');
            class_train = classify(train_dat_combined', train_dat_combined', ...
                train_labels_combined, 'linear');
            equal_test = class_test == test_labels_combined;
            equal_train = class_train == train_labels_combined;
            number_right_test = sum(equal_test(:) == 1);
            number_right_train = sum(equal_train(:) == 1);
            percent_right_test = number_right_test / (length(test_labels_combined));
            percent_right_train = number_right_train / (length(train_labels_combined));
            percent_correct_lda_three_digits_test = [percent_correct_lda_three_digits_test; ...
                [i, j, k, percent_right_test]];
            percent_correct_lda_three_digits_train = [percent_correct_lda_three_digits_train; ...
                [i, j, k, percent_right_train]];
           
            % Do SVM
            train_dat_combined_trans = train_dat_combined';
            mdl = fitcecoc((1 / max(train_dat_combined_trans(:))).* ...
                train_dat_combined', train_labels_combined);
            test_labels = predict(mdl, (1 / ...
                max(train_dat_combined_trans(:))).* test_dat_combined');
            train_labels = predict(mdl, (1 / ...
                max(train_dat_combined_trans(:))).* train_dat_combined');
            equal_svm_test = test_labels_combined == test_labels;
            equal_svm_train = train_labels_combined == train_labels;
            number_right_svm_test = sum(equal_svm_test(:) == 1);
            number_right_svm_train = sum(equal_svm_train(:) == 1);
            percent_right_svm_test = number_right_svm_test / (length(test_labels_combined));
            percent_right_svm_train = number_right_svm_train / (length(train_labels_combined));
            percent_correct_svm_three_digits_test = [percent_correct_svm_three_digits_test; ...
                [i, j, k, percent_right_svm_test]];
            percent_correct_svm_three_digits_train = [percent_correct_svm_three_digits_train; ...
                [i, j, k, percent_right_svm_train]];
            
            % Decision Tree Learning
            tree = fitctree(train_dat_combined', train_labels_combined);
            test_labels_tree = predict(tree, test_dat_combined');
            train_labels_tree = predict(tree, train_dat_combined');
            equal_tree_test = test_labels_tree == test_labels_combined;
            equal_tree_train = train_labels_tree == train_labels_combined;
            number_right_tree_test = sum(equal_tree_test(:) == 1);
            number_right_tree_train = sum(equal_tree_train(:) == 1);
            percent_right_tree_test = number_right_tree_test / (length(test_labels_combined));
            percent_right_tree_train = number_right_tree_train / (length(train_labels_combined));
            percent_correct_decis_tree_three_digits_test = [percent_correct_decis_tree_three_digits_test; ...
                [i, j, k, percent_right_tree_test]];
            percent_correct_decis_tree_three_digits_train = [percent_correct_decis_tree_three_digits_train; ...
                [i, j, k, percent_right_tree_train]];
        end
    end
end

% LDA - All Digits
class_test = classify(project_dat_test', project_dat', labels_train, 'linear');
class_train = classify(project_dat', project_dat', labels_train, 'linear');
equal_test = labels_test == class_test;
equal_train = labels_train == class_train;
number_right_test = sum(equal_test(:) == 1);
number_right_train = sum(equal_train(:) == 1);
percent_right_test = number_right_test / length(labels_test);
percent_right_train = number_right_train / length(labels_train);

% SVM - All Digits
project_dat_trans = project_dat';
mdl = fitcecoc((1 / max(project_dat_trans(:))).* project_dat', labels_train);
test_labels = predict(mdl, (1 / max(project_dat_trans(:))).* project_dat_test');
train_labels = predict(mdl, (1 / max(project_dat_trans(:))).* project_dat');
equal_svm_test = labels_test == test_labels;
equal_svm_train = labels_train == train_labels;
number_right_svm_test = sum(equal_svm_test(:) == 1);
number_right_svm_train = sum(equal_svm_train(:) == 1);
percent_right_svm_test = number_right_svm_test / length(labels_test);
percent_right_svm_train = number_right_svm_train / length(labels_train);

% Decision Tree Learning - All Digits
tree = fitctree(project_dat', labels_train);
test_labels_tree = predict(tree, project_dat_test');
train_labels_tree = predict(tree, project_dat');
equal_tree_test = test_labels_tree == labels_test;
equal_tree_train = train_labels_tree == labels_train;
number_right_tree_test = sum(equal_tree_test(:) == 1);
number_right_tree_train = sum(equal_tree_train(:) == 1);
percent_right_tree_test = number_right_tree_test / length(labels_test);
percent_right_tree_train = number_right_tree_train / length(labels_train);

