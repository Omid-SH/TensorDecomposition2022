% HW5 TD
% Auth : Omid Sharafi - 2022
% Git-Hub : https://github.com/Omid-SH

%% Start from here!
addpath('tensor_toolbox-v3.2.1')

%% 1

%% Test HOOI with small clear tensor
[U1, ~] = gsog(rand(7,5));
[U2, ~] = gsog(rand(8,7));
[U3, ~] = gsog(rand(10,4));

G = rand(5, 7, 4);
T = ttm(tensor(G), {U1, U2, U3}, 1:3);
rank = [3, 5, 3];
Itr = [0 1 10 100 1000];

for itr = Itr
    
    [G, U] = HOOI(T, rank, itr);

    T_pred = ttm(G, U, 1:3);
    error = sqrt(sum(double(T_pred-T).^2, 'all'));
    fprintf('Frobenius norm error = %d, Iteration = %d\n', error, itr);
    
end

%% Test HOOI with ‌Big clear tensor

[U1, ~] = gsog(rand(30,20));
[U2, ~] = gsog(rand(25,7));
[U3, ~] = gsog(rand(40,24));

G = rand(20, 7, 24);
T_org = ttm(tensor(G), {U1, U2, U3}, 1:3);
T = T_org + tensor(0.3*randn(30, 25, 40));
rank = [20, 7, 24];
Itr = 1:5;

for itr = Itr
    
    [G, U] = HOOI(T, rank, itr);

    T_pred = ttm(G, U, 1:3);
    error = sqrt(sum(double(T_pred-T).^2, 'all'));
    fprintf('Frobenius norm error = %d, Iteration = %d\n', error, itr);
    
end


%% 2

% Load Data
T = zeros([112, 92, 50]);

for i = 1:5
    for j = 1:10
        T(:, :, (i-1)*10+j) = imread(['ORL/s', num2str(i), '/', num2str(j), '.pgm']);
    end
end

% Draw Dataset

person = 3;
figure(person)
for j = 1:10
    subplot(2,5,j)
    imshow(uint8(squeeze(T(:, :, (person-1)*10+j))))
end

%% Our HOOI Output
rank = [50, 50, 5];
itr = 100;
[G, U] = HOOI(tensor(T), rank, itr);
plot(abs(U{3}));
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5');
grid on; 
title('HOOI Output (Abs)');

%% Tucker ALS
rank = [50, 50, 5];
[G, U] = tucker_als(tensor(T), rank);
plot(abs(U{3}));
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5');
grid on; 
title('Tucker ALS (Abs)');

%% Non-Negative Tucker Algorithm
opts.maxit = 700;
opts.tol = eps;
R = [50, 50, 5];
[A, C, Out] = ntd(tensor(T), R, opts);
plot(A{3});
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5');
grid on;
title('Non-Negative Tucker Algorithm');

%% 3

%% A)
scale = 0.5;
T = zeros([10, 9, 32256 * scale * scale]);
sub_name = {'+000E', '+010E', '+025E', '+050E', '+070E', '-010E', '-025E', '-050E', '-070E'};
% Read Data
for person = 1:10
    if person <10
        parent_name = ['0', num2str(person)];
    else
        parent_name = num2str(person);
    end
    for illum = 1:9
        temp = imresize(imread(['Illumination_Yale', '/', 'yaleB', parent_name, '/', 'yaleB', parent_name, ...
            '_P00A', sub_name{illum},'+00.pgm']), scale);
        T(person, illum, :) = temp(:);
    end
end

%% Tucker

R1 = 10;
R2 = 1:2:5;
R3 = 90;
prediction = cell(1, length(R2));
cnt = 1;

for r2 = R2
    temp = tucker_als(tensor(double(T)),[10 r2 90]);
    prediction{cnt} = ttm(temp.core,temp.U,1:3).data;
    cnt = cnt + 1;
end

%% Ploting
for person = 1:10
    figure(person);
    if person <10
        parent_name = ['0', num2str(person)];
    else
        parent_name = num2str(person);
    end
    for illum = 1:9
        for i = 1:4
            if i == 1 
                img = imresize(imread(['Illumination_Yale', '/', 'yaleB', parent_name, '/', 'yaleB', parent_name, ...
                '_P00A', sub_name{illum},'+00.pgm']), scale);
            else
                img = uint8(reshape(prediction{i-1}(person, illum, :), 96, 84));
            end
            subplot(4,9,(i-1)*9+illum)
            imshow(img)
            if illum == 1
                if (i==1)
                    ylabel('Original image')
                else
                    ylabel(['R = ', num2str(R2(i-1))])
                end
            end
            if i == 1
                title(num2str(sub_name{illum}))
            end
        end
    end
    sgtitle(['Person ', num2str(person)]) 
end

%% B)
X = reshape(tenmat(T, 2).data,[90, 8064]);

%% test order of matrix images
imshow(uint8(reshape(X(9,:), 96, 84)))

%% SVD

R = [10 30 50];
prediction = cell(1, length(R));
cnt = 1;

[U, S, V] = svd(X);

for r = R
    prediction{cnt} = U(:,1:r) * S(1:r, 1:r) * V(:, 1:r)';
    cnt = cnt + 1;
end

%% Ploting
for person = 1:10
    figure(person);
    if person <10
        parent_name = ['0', num2str(person)];
    else
        parent_name = num2str(person);
    end
    for illum = 1:9
        for i = 1:4
            if i == 1 
                img = imresize(imread(['Illumination_Yale', '/', 'yaleB', parent_name, '/', 'yaleB', parent_name, ...
                '_P00A', sub_name{illum},'+00.pgm']), scale);
            else
                img = uint8(reshape(prediction{i-1}((person-1)*9+illum, :), 96, 84));
            end
            subplot(4,9,(i-1)*9+illum)
            imshow(img)
            if illum == 1
                if (i==1)
                    ylabel('Original image')
                else
                    ylabel(['R = ', num2str(R(i-1))])
                end
            end
            if i == 1
                title(num2str(sub_name{illum}))
            end
        end
    end
    sgtitle(['Person ', num2str(person)]) 
end

%% Functions

function [G, U] = HOOI(T, rank, itr_n)

    % HOSVD for initial value
    U = cell(1, length(rank));
    for i = 1:length(rank)
        [U_temp, ~, ~] = svd(tenmat(T, i).data);
        U{i} = U_temp(:, 1:rank(i));
    end
    
    % Calculate U matrices using HOSVD and iteration
    for itr = 1:itr_n
        for k = 1:length(rank)
            temp_cell= cell(1,length(rank)-1);
            cnt = 1;
            for i = [1:k-1, k+1:length(rank)]
                temp_cell{cnt} = U{i}';
                cnt = cnt + 1;
            end
            [U_temp, ~, ~] = svd(tenmat(ttm(T, temp_cell, [1:k-1, k+1:length(rank)]),k).data);
            U{k} = U_temp(:, 1:rank(k));
        end
    end
    
    % Calculate Core Matrix
    temp_cell = cell(1,length(rank));
    for i = 1:length(rank)
        temp_cell{i} = U{i}';
    end
    G = ttm(T, temp_cell, 1:length(rank));
end

function [Q, R] = gsog(X)
    % Gram-Schmidt orthogonalization
    % Written by Mo Chen (sth4nth@gmail.com).
    [d,n] = size(X);
    m = min(d,n);
    R = eye(m,n);
    Q = zeros(d,m);
    D = zeros(1,m);
    for i = 1:m
        R(1:i-1,i) = bsxfun(@times,Q(:,1:i-1),1./D(1:i-1))'*X(:,i);
        Q(:,i) = X(:,i)-Q(:,1:i-1)*R(1:i-1,i);
        D(i) = dot(Q(:,i),Q(:,i));
    end
    R(:,m+1:n) = bsxfun(@times,Q,1./D)'*X(:,m+1:n);
end