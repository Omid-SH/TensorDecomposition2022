%% 
% NMF
% Auth : Omid Sharafi (2022) 
% https://github.com/Omid-SH

%% 1)

%% Check 1
A = rand(10, 6);

for j = 1:6
    
    B0 = rand(10, j);
    C0 = rand(j, 6);
    
    [B, C] = nmf_als(A, j, B0, C0, 100);
    e1 = norm(B * C - A, 'fro');
    [B, C] = nnmf(A, j, 'algorithm', 'als', 'w0', B0, 'h0', C0);
    e2 = norm(B * C - A, 'fro');
    [B, C] = nmf_mul(A, j, B0, C0, 100);
    e3 = norm(B * C - A, 'fro');
    [B, C] = nnmf(A, j, 'algorithm', 'mult', 'w0', B0, 'h0', C0);
    e4 = norm(B * C - A, 'fro');
    
    fprintf('Error J = %d \n- Our ALS : %d \n- Matlab ALS : %d\n- Our Mult : %d \n- Matlab Mult : %d \n\n',...
        j, e1, e2, e3, e4);    
end

%% Check 2

B = rand(6, 3);
C = rand(3, 4);
E = rand(6, 4);

snr = [-10, 0, 10, 30, 50];

A_stack = zeros(6,4,5);

for i = 1:5
    alpha = norm(B * C, 'fro') / 10 ^ (snr(i)/20) / norm(E, 'fro');
    A_stack(:, :, i) = B * C + alpha * E;
end

Error1 = zeros(5,3);
Error2 = zeros(5,3);
Error3 = zeros(5,3);
Error4 = zeros(5,3);

for i = 1:5
    for j = 1:4
        
        % make ten random initialization
        e1 = 0;
        e2 = 0;
        e3 = 0;
        e4 = 0;
        A = A_stack(:, :, i);

        for itr = 1:10
            B0 = rand(6, j);
            C0 = rand(j, 4);

            [B, C] = nmf_als(A, j, B0, C0, 100);
            e1 = e1 + norm(B * C - A, 'fro');
            [B, C] = nnmf(A, j, 'algorithm', 'als', 'w0', B0, 'h0', C0);
            e2 = e2 + norm(B * C - A, 'fro');
            [B, C] = nmf_mul(A, j, B0, C0, 100);
            e3 = e3 + norm(B * C - A, 'fro');
            [B, C] = nnmf(A, j, 'algorithm', 'mult', 'w0', B0, 'h0', C0);
            e4 = e4 + norm(B * C - A, 'fro');
        end
        
        Error1(i,j) = e1 / 10;
        Error2(i,j) = e2 / 10;
        Error3(i,j) = e3 / 10;
        Error4(i,j) = e4 / 10;

    end
end

for splot = 1:4
    subplot(2,2,splot)
    plot(snr, Error1(:,splot))
    hold on
    plot(snr, Error2(:,splot))
    plot(snr, Error3(:,splot))
    plot(snr, Error4(:,splot))
    legend('Our ALS', 'Matlab ALS', 'Our Mult', 'Matlab Mult')
    xlabel('Noise SNR(10dB)')
    ylabel('Error')
    title(['J = ', num2str(splot)])
    hold off
end


%% 2
clear
load('swimmer.mat');

%% solve using Mult

Y = zeros(length(A), size(A{1},1) * size(A{1},2));
for i = 1:length(A)
    Y(i, :) = reshape(A{i}, 1, size(A{1},1) * size(A{1},2));
end

error = zeros(1,126);
for j = 1:126
    [B, C] = nnmf(Y, j, 'algorithm', 'mult');
    error(j) = norm(Y - B * C, 'fro');
end

figure;
plot(1:126, error);
title('mult algorithm');
grid on;
xlabel('j');
ylabel('Error');

%% 16 is best for Mul
[B, C] = nnmf(Y, 16, 'algorithm', 'mult');
for i = 1:16
    subplot(4,4,i)
    imagesc(reshape(C(i,:), 9, 14))
end

%% test for j = 9
[B, C] = nnmf(Y, 9, 'algorithm', 'mult');
for i = 1:9
    subplot(3,3,i)
    imagesc(reshape(C(i,:), 9, 14))
end

%% test for j = 24
[B, C] = nnmf(Y, 24, 'algorithm', 'mult');
for i = 1:24
    subplot(4,6,i)
    imagesc(reshape(C(i,:), 9, 14))
end

%% solve using ALS

Y = zeros(length(A), size(A{1},1) * size(A{1},2));
for i = 1:length(A)
    Y(i, :) = reshape(A{i}, 1, size(A{1},1) * size(A{1},2));
end

error = zeros(1,19);
for j = 1:19
    [B, C] = nnmf(Y, j, 'algorithm', 'als');
    error(j) = norm(Y - B * C, 'fro');
end

figure;
plot(1:19, error);
title('ALS algorithm');
grid on;
xlabel('j');
ylabel('Error');

%% 10 is best for ALS
[B, C] = nnmf(Y, 10, 'algorithm', 'als');
for i = 1:10
    subplot(2,5,i)
    imagesc(reshape(C(i,:), 9, 14))
end

%% test for j = 4
[B, C] = nnmf(Y, 4, 'algorithm', 'als');
for i = 1:4
    subplot(2,2,i)
    imagesc(reshape(C(i,:), 9, 14))
end

%% test for j = 24
[B, C] = nnmf(Y, 24, 'algorithm', 'als');
for i = 1:24
    subplot(4,6,i)
    imagesc(reshape(C(i,:), 9, 14))
end
%% Functions

function [B, C] = nmf_als(A, j, B0, C0, itr)

    B = B0;
    C = C0;
    
    for i = 1:itr
         B = max(eps, (A * C') * pinv(C * C'));
         C = max(eps, pinv(B' * B)*(B' * A));
    end

end

function [B, C] = nmf_mul(A, j, B0, C0, itr)

    B = B0;
    C = C0;
    
    for i = 1:itr
        B = B .* (A * C') ./ (B * C * C' + ones(size(B)) * eps);
        C = C .* (B' * A) ./ (B' * B * C + ones(size(C)) * eps);
    end

end
