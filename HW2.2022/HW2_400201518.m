%% 
% SVD and PCA
% Auth : Omid Sharafi (2022) 
% https://github.com/Omid-SH

%% 1)
img = imread('cameraman.tif');
img = im2double(img);
img_zm = img - mean(img);
im_norm = normalize(img);


%% A)

error = zeros(1,9);
[U, S, V] = svd(img);
figure('Name', 'Original Image')
for i=1:9
    subplot(3,3,i)
    k = 2^(i-1);
    img_temp = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';
    imshow(img_temp)
    title(['K = ', num2str(k)])
    error(i) = norm(img-img_temp, 'fro')/norm(img, 'fro');
end

error_zm = zeros(1,9);
[U, S, V] = svd(img_zm);
figure('Name', 'Zero Mean Image')
for i=1:9
    subplot(3,3,i)
    k = 2^(i-1);
    img_temp = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)' + mean(img);
    imshow(img_temp)
    title(['K = ', num2str(k)])
    error_zm(i) = norm(img-img_temp, 'fro')/norm(img, 'fro');
end

error_norm = zeros(1,9);
[U, S, V] = svd(im_norm);
figure('Name', 'Normalized Image')
for i=1:9
    subplot(3,3,i)
    k = 2^(i-1);
    img_temp = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';
    imshow(img_temp)
    title(['K = ', num2str(k)])
    error_norm(i) = norm(im_norm-img_temp, 'fro')/norm(im_norm, 'fro');
end

%% B)
figure('Name', 'Error')
subplot(1,3,1)
plot(2.^(0:8), error*100, '-s')
title('Error percentage based on K using Orginal Image')
ylim([0 80])
xlim([1 256])
subplot(1,3,2)
plot(2.^(0:8), error_zm*100, '-s')
title('Error percentage based on K using Zero meaned Image')
ylim([0 80])
xlim([1 256])
subplot(1,3,3)
plot(2.^(0:8), error_norm*100, '-s')
title('Error percentage based on K using Normalized Image')
ylim([0 80])
xlim([1 256])

%% C)
figure('Name', 'Error')
subplot(1,3,1)
plot((256*2+1).*2.^(0:8)/256/256, error*100, '-s')
title('Error percentage based on compresion rate using Orginal Image')
ylim([0 80])
subplot(1,3,2)
plot((256*2+1).*2.^(0:8)/256/256, error_zm*100, '-s')
title('Error percentage based on compresion rate using Zero meaned Image')
ylim([0 80])
subplot(1,3,3)
plot((256*2+1).*2.^(0:8)/256/256, error_norm*100, '-s')
title('Error percentage based on compresion rate using Normalized Image')
ylim([0 80])

%% 2)

load('EEGdata.mat');
Xorg = Xorg';
Xnoise = Xnoise';
[U, S, V] = svd(Xnoise-mean(Xnoise));

source = U*S;
for k=1:32
    subplot(4, 8, k);
    plot(source(:, k));
    title(['Channel ', num2str(k)]);
end

error = zeros(1,32);
for k=1:32
    sig = U(:, k)*S(k, k)*V(:, k)'+ mean(Xnoise);
    error(k) = norm(Xorg-sig, 'fro');
end

figure()
stem(1:32, error);
xlabel('K')
title('Frobenius error between X_{org} and X_{reconst}');


error = zeros(1,32);
for k=1:32
    sig = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)'+ mean(Xnoise);
    error(k) = norm(Xorg-sig, 'fro');
end

figure()
stem(1:32, error);
xlabel('1:K')
title('Frobenius error between X_{org} and X_{reconst}');

error = zeros(32,32) + diag(error);
for k=1:32
    for j=k+1:32
        sig = U(:, [k j])*S([k j], [k j])*V(:, [k j])'+ mean(Xnoise);
        error(k,j) = norm(Xorg-sig, 'fro');
        error(j,k) = error(k,j);
    end
end

figure()
heatmap(error)
title('Frobenius error between X_{org} and X_{reconst} using i and j sources');

%% 3)

load('PCAdata.mat');
PCAdata = PCAdata';

%% A)
figure()
scatter3(PCAdata(:, 1), PCAdata(:, 2), PCAdata(:, 3), 15, 'filled')

figure()
scatter(PCAdata(:, 1), PCAdata(:, 2), 15, 'filled')
xlabel('X')
ylabel('Y')

figure()
scatter(PCAdata(:, 1), PCAdata(:, 3), 15, 'filled')
xlabel('X')
ylabel('Z')

figure()
scatter(PCAdata(:, 2), PCAdata(:, 3), 15, 'filled')
xlabel('Y')
ylabel('Z')

x_tautness = norm(PCAdata(:, 1))
y_tautness = norm(PCAdata(:, 2))
z_tautness = norm(PCAdata(:, 3))

PCAdata_zm = PCAdata - mean(PCAdata);

figure()
scatter3(PCAdata_zm(:, 1), PCAdata_zm(:, 2), PCAdata_zm(:, 3), 15, 'filled')

figure()
scatter(PCAdata_zm(:, 1), PCAdata_zm(:, 2), 15, 'filled')
xlabel('X')
ylabel('Y')

figure()
scatter(PCAdata_zm(:, 1), PCAdata_zm(:, 3), 15, 'filled')
xlabel('X')
ylabel('Z')

figure()
scatter(PCAdata_zm(:, 2), PCAdata_zm(:, 3), 15, 'filled')
xlabel('Y')
ylabel('Z')

x_tautness_zm = norm(PCAdata_zm(:, 1))
y_tautness_zm = norm(PCAdata_zm(:, 2))
z_tautness_zm = norm(PCAdata_zm(:, 3))


%% B)
[U, S, V] = svd(PCAdata_zm);
tautness_svd = S;
variance_svd = S.^2/1000;
direction_svd = V;
wdata_svd = U*S;

%% C)
[coeff, score, latent] = pca(PCAdata_zm);
tautness_pca = sqrt(latent*1000);
variance_pca = latent;
direction_pca = coeff;
wdata_pca = score;

%% 4)
load('house_dataset.mat');

houseInputs = houseInputs';
houseTargets = houseTargets';

% houseInputs = houseInputs - mean(houseInputs);
% [U, S, V] = svd(houseInputs);
% source = U*S;

houseInputs = normalize(houseInputs);
[U, S, V] = svd(houseInputs);
source = U*S;

covariance = zeros(1, 13);
for i=1:13
    covariance(i) = (dot(houseTargets, source(:, i)) - mean(houseTargets) * mean(source(:, i)))/norm(houseTargets)/norm(source(:, i));
end


corr = zeros(13,13) + diag(abs(covariance));
for k=1:13
    for j=k+1:13
        sig = sum(U(:, [k j])*S([k j], [k j]), 2);
        corr(k,j) = abs((dot(houseTargets, sig) - mean(houseTargets) * mean(sig))/norm(houseTargets)/norm(sig));
        corr(j,k) = corr(k,j);
    end
end

figure()
heatmap(corr)
title('Abs of Correlation between price and combination feature using i and j sources');


figure();
scatter(source(:, 1), source(:, 5), 30, houseTargets, 'filled');
xlabel('First source');
ylabel('Fifth source');
title('House price')
colorbar;