% TD HW4 programming Tensor CP Algorithms
% Auth : Omid Sharafi 2022
% Github : https://github.com/Omid-SH

%% 2
addpath('tensorlab_2016-03-28');

%% 2-A

Errors = zeros([50, 4, 4]);
% 50 tensor, 4 noise, 4 algorithms(ALS, cpd_als, cpd3_sd, cpd_minf)

for cnt = 1:50
    
    T = zeros(6, 6, 6);
    
    U1_org = randn(6, 3);
    U2_org = randn(6, 3);
    U3_org = randn(6, 3);
    U_org = {U1_org, U2_org, U3_org};
    
    for r = 1:3
        A = U1_org(:,r);
        B = U2_org(:,r);
        C = U3_org(:,r);
        
        % Outer product
        AB = reshape(A(:) * B(:).', [size(A, 1), size(B, 1)]);
        T = T + reshape(AB(:) * C(:).', [size(AB), size(C)]);
    end
    
    SNR = [0, 20, 40, 60];
    for snr = 1:4
        
        % Add noise
        N = randn(6, 6, 6);
        alpha = tensor_norm_fro(T) / tensor_norm_fro(N) / 10^(SNR(snr)/20);
        T_noisy = T + alpha * N;
        
        % Make initial components Using 
        U1_0 = randn(6, 3);
        U2_0 = randn(6, 3);
        U3_0 = randn(6, 3);
        U_0 = {U1_0, U2_0, U3_0}; 
        
        % ALS
        [U1, U2, U3] = ALS(T_noisy, U1_0, U2_0, U3_0, 100);
        Errors(cnt, snr, 1) = TMSFE(U_org, {U1, U2, U3});
        
        % cpd_als
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_als);
        Errors(cnt, snr, 2) = TMSFE(U_org, U);
        
        % cpd3_sd
        U = cpd3_sd(T_noisy, U_0);
        Errors(cnt, snr, 3) = TMSFE(U_org, U);
        
        % cpd_minf
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_minf);
        Errors(cnt, snr, 4) = TMSFE(U_org, U);
        
    end
    
    fprintf('Iteration %d/ %d\n', cnt, 50)
    
end

%% Plot the result
Error_mean = squeeze(mean(Errors));
figure()
plot(SNR, Error_mean(:, 1));
hold on;
plot(SNR, Error_mean(:, 2));
plot(SNR, Error_mean(:, 3));
plot(SNR, Error_mean(:, 4));

xlabel('SNR');
ylabel('Error');
title('CP Error (Meaned Over 50 different tensors), Initialized Randomly');
legend('My ALS', 'CPD ALS', 'CPD3 SD', 'CPD MINF');

%% 2-B

Errors = zeros([50, 4, 4]);
% 50 tensor, 4 noise, 4 algorithms(ALS, cpd_als, cpd3_sd, cpd_minf)

for cnt = 1:50
    
    T = zeros(6, 6, 6);
    
    U1_org = randn(6, 3);
    U2_org = randn(6, 3);
    U3_org = randn(6, 3);
    U_org = {U1_org, U2_org, U3_org};
    
    for r = 1:3
        A = U1_org(:,r);
        B = U2_org(:,r);
        C = U3_org(:,r);
        
        % Outer product
        AB = reshape(A(:) * B(:).', [size(A, 1), size(B, 1)]);
        T = T + reshape(AB(:) * C(:).', [size(AB), size(C)]);
    end
    
    SNR = [0, 20, 40, 60];
    for snr = 1:4
        
        % Add noise
        N = randn(6, 6, 6);
        alpha = tensor_norm_fro(T) / tensor_norm_fro(N) / 10^(SNR(snr)/20);
        T_noisy = T + alpha * N;
        
        % Make initial components
        [U1_0, U2_0, U3_0] =  HOSVD(T_noisy);
        U_0 = {U1_0, U2_0, U3_0};
        
        % ALS
        [U1, U2, U3] = ALS(T_noisy, U1_0, U2_0, U3_0, 100);
        Errors(cnt, snr, 1) = TMSFE(U_org, {U1, U2, U3});
        
        % cpd_als
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_als);
        Errors(cnt, snr, 2) = TMSFE(U_org, U);
        
        % cpd3_sd
        U = cpd3_sd(T_noisy, U_0);
        Errors(cnt, snr, 3) = TMSFE(U_org, U);
        
        % cpd_minf
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_minf);
        Errors(cnt, snr, 4) = TMSFE(U_org, U);
        
    end
    
    fprintf('Iteration %d/ %d\n', cnt, 50)
    
end

%% Plot the result
Error_mean = squeeze(mean(Errors));
figure()
plot(SNR, Error_mean(:, 1));
hold on;
plot(SNR, Error_mean(:, 2));
plot(SNR, Error_mean(:, 3));
plot(SNR, Error_mean(:, 4));

xlabel('SNR');
ylabel('Error');
title('CP Error (Meaned Over 50 different tensors), Initialized Using HOSVD');
legend('My ALS', 'CPD ALS', 'CPD3 SD', 'CPD MINF');

%% 2-C

Errors = zeros([50, 4, 4]);
% 50 tensor, 4 noise, 4 algorithms(ALS, cpd_als, cpd3_sd, cpd_minf)

for cnt = 1:50
    
    T = zeros(6, 6, 6);
    
    c1 = randn(6, 1);
    c2 = c1 + 0.5*randn(6, 1);
    c3 = randn(6, 1);
    U1_org = [c1 c2 c3];
    
    c1 = randn(6, 1);
    c2 = c1 + 0.5*randn(6, 1);
    c3 = randn(6, 1);
    U2_org = [c1 c2 c3];
    
    U3_org = randn(6, 3);
    U_org = {U1_org, U2_org, U3_org};
    
    for r = 1:3
        A = U1_org(:,r);
        B = U2_org(:,r);
        C = U3_org(:,r);
        
        % Outer product
        AB = reshape(A(:) * B(:).', [size(A, 1), size(B, 1)]);
        T = T + reshape(AB(:) * C(:).', [size(AB), size(C)]);
    end
    
    SNR = [0, 20, 40, 60];
    for snr = 1:4
        
        % Add noise
        N = randn(6, 6, 6);
        alpha = tensor_norm_fro(T) / tensor_norm_fro(N) / 10^(SNR(snr)/20);
        T_noisy = T + alpha * N;
        
        % Make initial components Using 
        U1_0 = randn(6, 3);
        U2_0 = randn(6, 3);
        U3_0 = randn(6, 3);
        U_0 = {U1_0, U2_0, U3_0}; 
        
        % ALS
        [U1, U2, U3] = ALS(T_noisy, U1_0, U2_0, U3_0, 100);
        Errors(cnt, snr, 1) = TMSFE(U_org, {U1, U2, U3});
        
        % cpd_als
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_als);
        Errors(cnt, snr, 2) = TMSFE(U_org, U);
        
        % cpd3_sd
        U = cpd3_sd(T_noisy, U_0);
        Errors(cnt, snr, 3) = TMSFE(U_org, U);
        
        % cpd_minf
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_minf);
        Errors(cnt, snr, 4) = TMSFE(U_org, U);
        
    end
    
    fprintf('Iteration %d/ %d\n', cnt, 50)
    
end

%% Plot the result
Error_mean = squeeze(mean(Errors));
figure()
plot(SNR, Error_mean(:, 1));
hold on;
plot(SNR, Error_mean(:, 2));
plot(SNR, Error_mean(:, 3));
plot(SNR, Error_mean(:, 4));

xlabel('SNR');
ylabel('Error');
title('CP Error For Correlated Base Matrix (Meaned Over 50 different tensors), Initialized Randomly');
legend('My ALS', 'CPD ALS', 'CPD3 SD', 'CPD MINF');

%% 2-D

Errors = zeros([50, 4, 4]);
% 50 tensor, 4 noise, 4 algorithms(ALS, cpd_als, cpd3_sd, cpd_minf)

for cnt = 1:50
    
    T = zeros(6, 6, 6);
    
    c1 = randn(6, 1);
    c2 = c1 + 0.5*randn(6, 1);
    c3 = randn(6, 1);
    U1_org = [c1 c2 c3];
    
    c1 = randn(6, 1);
    c2 = c1 + 0.5*randn(6, 1);
    c3 = randn(6, 1);
    U2_org = [c1 c2 c3];
    
    U3_org = randn(6, 3);
    U_org = {U1_org, U2_org, U3_org};
    
    for r = 1:3
        A = U1_org(:,r);
        B = U2_org(:,r);
        C = U3_org(:,r);
        
        % Outer product
        AB = reshape(A(:) * B(:).', [size(A, 1), size(B, 1)]);
        T = T + reshape(AB(:) * C(:).', [size(AB), size(C)]);
    end
    
    SNR = [0, 20, 40, 60];
    for snr = 1:4
        
        % Add noise
        N = randn(6, 6, 6);
        alpha = tensor_norm_fro(T) / tensor_norm_fro(N) / 10^(SNR(snr)/20);
        T_noisy = T + alpha * N;
        
        % Make initial components
        [U1_0, U2_0, U3_0] =  HOSVD(T_noisy);
        U_0 = {U1_0, U2_0, U3_0};
        
        % ALS
        [U1, U2, U3] = ALS(T_noisy, U1_0, U2_0, U3_0, 100);
        Errors(cnt, snr, 1) = TMSFE(U_org, {U1, U2, U3});
        
        % cpd_als
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_als);
        Errors(cnt, snr, 2) = TMSFE(U_org, U);
        
        % cpd3_sd
        U = cpd3_sd(T_noisy, U_0);
        Errors(cnt, snr, 3) = TMSFE(U_org, U);
        
        % cpd_minf
        U = cpd(T_noisy, U_0, 'Algorithm', @cpd_minf);
        Errors(cnt, snr, 4) = TMSFE(U_org, U);
        
    end
    
    fprintf('Iteration %d/ %d\n', cnt, 50)
    
end

%% Plot the result
Error_mean = squeeze(mean(Errors));
figure()
plot(SNR, Error_mean(:, 1));
hold on;
plot(SNR, Error_mean(:, 2));
plot(SNR, Error_mean(:, 3));
plot(SNR, Error_mean(:, 4));

xlabel('SNR');
ylabel('Error');
title('CP Error For Correlated Base Matrix (Meaned Over 50 different tensors), Initialized Using HOSVD');
legend('My ALS', 'CPD ALS', 'CPD3 SD', 'CPD MINF');

%% 3
load amino.mat
T = reshape(X,[5,201,61]);

%% 3-A
Out = cell([7, 5]);
% 7 algorithms, 5 R

for R = 2:5
    
    % Make initial components Using 
    U1_0 = randn(5, R);
    U2_0 = randn(201, R);
    U3_0 = randn(61, R);
    U_0 = {U1_0, U2_0, U3_0}; 

    % ALS
    [U1, U2, U3] = ALS(T, U1_0, U2_0, U3_0, 100);
    Out{1, R} = {U1, U2, U3};

    % cpd_als
    Out{2, R} = cpd_als(T, U_0);

    % cpd3_sd
    Out{3, R} = cpd3_sd(T, U_0);

    % cpd_minf
    Out{4, R} = cpd_minf(T, U_0);

    % cpd3_sgsd
    Out{5, R} = cpd3_sgsd(T, U_0);

    % cpd_core
    Out{6, R} = cpd_core(T, U_0);

    % cpd_nls
    Out{7, R} = cpd_nls(T, U_0);
    
    fprintf('Process %d/ %d\n', R-1, 4)
    
end

%% Plot the result
method_names = {'My ALS', 'CPD ALS', 'CPD3 SD', 'CPD MINF', 'CPD3 SGSD', 'CPD CORE', 'CPD NLS'};

for R = 2:5
    figure()
    % Plot emissions
    for method = 1:7
        subplot(4,2,method)
        plot(EmAx', Out{method, R}{2})
        title(['Emission using ', method_names{method}, ' method, R = ', num2str(R)])
    end
    
    figure()
    % Plot exitation
    for method = 1:7
        subplot(4,2,method)
        plot(ExAx', Out{method, R}{3})
        title(['Excitation using ', method_names{method}, ' method, R = ', num2str(R)])
    end
end

%% 3-B
addpath('corcond')
for R = 2:5
    for method = 1:7
        figure()
        [Consistency,~,~,~] = corcond(T, Out{method, R}, [], 1);
        title(['Corcond of ', method_names{method}, ' method, R = ', num2str(R), ', Core Consistency: ', num2str(Consistency)]);
        grid on;
    end
end

%% Functions

function [U1, U2, U3] = ALS(T, U1_0, U2_0, U3_0, n_itr)

    % Unfold-mode1
    Un1 = zeros(size(T,1), size(T,2)*size(T,3));
    cnt = 1;
    for k = 1:size(T,3)
        for j = 1:size(T,2)
            Un1(:, cnt) = T(:, j, k);
            cnt = cnt + 1;
        end
    end
    
    % Unfold-mode2
    Un2 = zeros(size(T,2), size(T,1)*size(T,3));
    cnt = 1;
    for k = 1:size(T,3)
        for i = 1:size(T,1)
            Un2(:, cnt) = T(i, :, k);
            cnt = cnt + 1;
        end
    end
    
    % Unfold-mode3
    Un3 = zeros(size(T,3), size(T,1)*size(T,2));
    cnt = 1;
    for j = 1:size(T,2)
        for i = 1:size(T,1)
            Un3(:, cnt) = T(i, j, :);
            cnt = cnt + 1;
        end
    end
    
    % ALS
    U1 = U1_0;
    U2 = U2_0;
    U3 = U3_0;
    for i = 1:n_itr
        U1 = Un1 * kr(U3,U2) * pinv((U2'*U2) .* (U3'*U3));
        U2 = Un2 * kr(U3,U1) * pinv((U1'*U1) .* (U3'*U3));
        U3 = Un3 * kr(U2,U1) * pinv((U1'*U1) .* (U2'*U2));
    end

end

function [U1, U2, U3] = HOSVD(T)

    % Unfold-mode1
    Un1 = zeros(size(T,1), size(T,2)*size(T,3));
    cnt = 1;
    for k = 1:size(T,3)
        for j = 1:size(T,2)
            Un1(:, cnt) = T(:, j, k);
            cnt = cnt + 1;
        end
    end
    
    % Unfold-mode2
    Un2 = zeros(size(T,2), size(T,1)*size(T,3));
    cnt = 1;
    for k = 1:size(T,3)
        for i = 1:size(T,1)
            Un2(:, cnt) = T(i, :, k);
            cnt = cnt + 1;
        end
    end
    
    % Unfold-mode3
    Un3 = zeros(size(T,3), size(T,1)*size(T,2));
    cnt = 1;
    for j = 1:size(T,2)
        for i = 1:size(T,1)
            Un3(:, cnt) = T(i, j, :);
            cnt = cnt + 1;
        end
    end
    
    % HOSVD
    [U1, ~, ~] = svd(Un1);
    [U2, ~, ~] = svd(Un2);
    [U3, ~, ~] = svd(Un3);
    U1 = U1(:, 1:3);
    U2 = U2(:, 1:3);
    U3 = U3(:, 1:3);

end

function nf = tensor_norm_fro (T)
    nf = 0;
    for i = 1:size(T,1)
        for j = 1:size(T,2)
            for k = 1:size(T,3)
                nf = nf + T(i,j,k) * T(i,j,k)';
            end 
        end
    end
end