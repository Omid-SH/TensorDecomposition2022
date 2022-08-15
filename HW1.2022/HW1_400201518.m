%% 
% Jacobi_eig, Jacobi_svd_2sided, Jacobi_svd_1sided
% Auth : Omid Sharafi (2022) 
% https://github.com/Omid-SH

%% Test Jacobi_eig
A = rand(6);
A = A*A'

tic
[jV, jD] = Jacobi_eig(A)
toc

tic
[V, D] = eig(A)
toc

%% Test Jacobi_svd_2sided

A = rand([4, 3])

tic
[jU1, jS1, jV1] = Jacobi_svd_2sided(A)
toc

tic
[U, S, V] = svd(A)
toc

%% Test Jacobi_svd_1sided

A = rand([5, 3])

tic
[jU1, jS1, jV1] = Jacobi_svd_1sided(A)
toc

jU1*jS1*jV1'

tic
[U, S, V] = svd(A)
toc

%% Functions

%% Jacobi_eig
function [jV, jD] = Jacobi_eig(A)

    delta = 0.0001 * norm(A, 'fro');

    jV = eye(size(A, 1));
    jD = A;
    
    while (off(jD) > delta)
        
%     method 1
        for p = 1:size(A, 1) - 1
            for q = p+1:size(A, 1)

                [c, s] = symSchur2(jD, p, q);

                J = eye(size(A, 1));
                J(p,p) = c;
                J(p,q) = s;
                J(q,p) = -s;
                J(q,q) = c;

                jV = jV*J;
                jD = J'*jD*J;

            end
        end

%     % method 2
%         [p,q] = find(abs(jD) == max(abs(jD - diag(diag(jD))),[],'all'), 1);
%         if (p > q) 
%            temp = p;
%            p = q;
%            q = temp;
%         end
%         [c, s] = symSchur2(jD, p, q);
% 
%         J = eye(size(A, 1));
%         J(p,p) = c;
%         J(p,q) = s;
%         J(q,p) = -s;
%         J(q,q) = c;
% 
% 
%         jV = jV * J;
%         jD = J' * jD * J;
%         
     end
    
    % sort eigenvalues
    jD = diag(jD);
    for i=1:size(A, 1) - 1
        for j=1:size(A, 1) - i
            if jD(j) > jD(j+1)
                % change eigenvalues
                temp = jD(j);
                jD(j) = jD(j+1);
                jD(j+1) = temp;
                
                % change eigenvectors
                temp = jV(:, j);
                jV(:, j) = jV(:, j+1);
                jV(:, j+1) = temp;
            end
        end
    end
    jD = diag(jD);
    
end

%% Jacobi_svd_2sided
function [jU2, jS2, jV2] = Jacobi_svd_2sided(A)

    delta = 0.0001 * norm(A, 'fro');
    
    [m, n] = size(A);
    jS2 = A;
    jU2 = eye(m);
    jV2 = eye(n);
    
    while (off(jS2) > delta)
        for p = 1:min(m, n)-1
            for q = p+1:min(m, n)
                
                [c1, s1, c2, s2] = asymSchur2(jS2, p, q);
                
                J1 = eye(m);
                J1(p,p) = c1;
                J1(p,q) = s1;
                J1(q,p) = -s1;
                J1(q,q) = c1;                

                J2 = eye(n);
                J2(p,p) = c2;
                J2(p,q) = s2;
                J2(q,p) = -s2;
                J2(q,q) = c2;    
                
                jS2 = J1' * jS2 * J2;
                jU2 = jU2 * J1;
                jV2 = jV2 * J2;
                
            end
        end
        
        if m < n
            % make all n-m end columns zero
            for p = 1:m
                for q = m+1:n
                    
                    if jS2(p, p) == 0
                        c2 = 0;
                        s2 = 1;
                    else
                        t = -jS2(p, q)/jS2(p, p);
                        c2 = 1/sqrt(1+t^2);
                        s2 = t*c2;
                    end
                    
                    J2 = eye(n);
                    J2(p,p) = c2;
                    J2(p,q) = s2;
                    J2(q,p) = -s2;
                    J2(q,q) = c2;   
                    
                    jS2 = jS2 * J2;
                    jV2 = jV2 * J2;
                    
                end
            end
        elseif m > n
            % make all m-n end rows zero
            for p = n + 1:m
                for q = 1:n
                    
                    if jS2(q, q) == 0
                        c1 = 0;
                        s1 = 1;
                    else
                        t = -jS2(p, q)/jS2(q, q);
                        c1 = 1/sqrt(1+t^2);
                        s1 = t*c1;
                    end
                    
                    J1 = eye(m);
                    J1(p,p) = c1;
                    J1(p,q) = s1;
                    J1(q,p) = -s1;
                    J1(q,q) = c1;      
                    
                    jS2 = J1' * jS2;
                    jU2 = jU2 * J1;

                end
            end   
        end
    end
    
    % make all coef positive
    for i=1:min(m, n)
        if jS2(i, i) < 0
            jS2(i, i) = -jS2(i, i);
            jU2(:, i) = -jU2(:, i);
        end
    end
    
    % sort vectors
    jS2 = diag(jS2);
    for i = 1:min(m, n)-1
        for j = 1:min(m, n)-i
            
            if jS2(j) < jS2(j+1)
                % change coef
                temp = jS2(j);
                jS2(j) = jS2(j+1);
                jS2(j+1) = temp;
                
                % swap vectors
                temp = jU2(:, j);
                jU2(:, j) = jU2(:, j+1);
                jU2(:, j+1) = temp;
                temp = jV2(:, j);
                jV2(:, j) = jV2(:, j+1);
                jV2(:, j+1) = temp;
            end
            
        end
    end
    
    temp = jS2;
    jS2 = zeros(m, n);
    for i=1:min(m, n)
        jS2(i, i) = temp(i);
    end
    
end

%% Jacobi_svd_1sided
function [jU1, jS1, jV1] = Jacobi_svd_1sided(A)

    [m, n] = size(A);

    if m <= n   
        
        delta = 0.0001 * norm(A*A', 'fro');
        D = A';
        jV1 = eye(m);
        
        % Just work on m columns
        while (off(D'*D) > delta)         
            for p = 1:m-1
                for q = p+1:m
                    
                    [c, s] = orthogonalization(D(:, p), D(:, q));
                    
                    J = eye(m);
                    J(p,p) = c;
                    J(p,q) = s;
                    J(q,p) = -s;
                    J(q,q) = c;
                    
                    D = D * J;
                    jV1 = jV1 * J;
                    
                end
            end         
        end
        
        % get out JU1 & JS1 from D
        jS1 = zeros(n, m);
        jU1 = zeros(n, m);
        for i = 1:m
            jS1(i, i) = norm(D(:, i));
            jU1(:, i) = D(:, i)/norm(D(:, i));
        end
        
        % Transpose everything to get final result
        temp = jU1;
        jU1 = jV1;
        jS1 = jS1';
        jV1 = temp;
        
    else
        
        delta = 0.0001 * norm(A'*A, 'fro');
        D = A;
        jV1 = eye(n);
        
        % Just work on n columns
        while (off(D'*D) > delta)
            for p=1:n-1
                for q=p+1:n
                    
                    [c, s] = orthogonalization(D(:, p), D(:, q));
                    
                    J = eye(n);
                    J(p,p) = c;
                    J(p,q) = s;
                    J(q,p) = -s;
                    J(q,q) = c;
                    
                    D = D * J;
                    jV1 = jV1 * J;
                    
                end
            end
        end
        
        % get out JU1 & JS1 from D
        jS1 = zeros(m, n);
        jU1 = zeros(m, n);
        for i = 1:n
            jS1(i, i) = norm(D(:, i));
            jU1(:, i) = D(:, i)/norm(D(:, i));
        end
        
    end
    
    
    jS1 = diag(jS1); 
    
    for i = 1:min(m, n)-1
        for j = 1:min(m, n)-i
            
            if jS1(j) < jS1(j+1)
                
                % change coef
                temp = jS1(j);
                jS1(j) = jS1(j+1);
                jS1(j+1) = temp;
                
                % swap vectors
                temp = jU1(:, j);
                jU1(:, j) = jU1(:, j+1);
                jU1(:, j+1) = temp;              
                temp = jV1(:, j);
                jV1(:, j) = jV1(:, j+1);
                jV1(:, j+1) = temp;
                
            end
            
        end
    end   
    
    jS1 = diag(jS1);

end

%% off
function out = off(A)

    out = norm(A, 'fro') ^ 2;
    for i = 1:min(size(A))
        out = out - A(i, i) ^ 2;
    end
    
end

%% symSchur2
function [c, s] = symSchur2(A, p, q)

    if (A(p,q) == 0)
        c = 1;
        s = 0;
    else
        t = (A(q,q)-A(p,p))/(2*A(p,q));
        if (t >= 0)
            t_min = 1/(t+sqrt(1+t^2));
        else
            t_min = 1/(t-sqrt(1+t^2));
        end
        c = 1/(sqrt(1+t_min^2));
        s = t_min * c;
    end
    
end

%% asymSchur2
function [c1, s1, c2, s2] = asymSchur2(A, p, q)

    if (A(p, q) == A(q, p))
        c = 1; 
        s = 0;
    else
        t = (A(q, p)-A(p, q))/(A(p, p)+A(q, q));
        c = 1/(sqrt(1+t^2));
        s = t*c;
    end
    
    temp = [c, s; -s, c] * A([p,q], [p,q]);
    [c2, s2] = symSchur2(temp, 1, 2);
    c1 = c * c2 + s * s2;
    s1 = c * s2 - s * c2;
    
end

%% orthogonalization
function [c, s] = orthogonalization(x, y)

    if (norm(x) == norm(y))
        c = 1/sqrt(2);
        s = 1/sqrt(2);
    else
        t = 2*x'*y/(norm(y)^2-norm(x)^2);
        c = sqrt((1+1/sqrt(1+t^2))/2);
        s = sqrt((1-1/sqrt(1+t^2))/2);
    end
    
end