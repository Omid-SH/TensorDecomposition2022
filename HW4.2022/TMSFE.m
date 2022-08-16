function tmsfe = TMSFE(U,esU)

N = length(U) ; %N-way array
P = size(U{1},2) ; %Rank

permVec = perms(1:P) ;

tmsfe = 0 ;
for n=1:N
    A = U{n} ;
    esA = esU{n} ;
    
    tempdist = [] ;
    for i = 1: size(permVec,1)
        newA = esA(:,permVec(i,:)) ;
        diffA = [] ;
        for p = 1:P
            diffA(:,p) = A(:,p) - newA(:,p)'*A(:,p)/(newA(:,p)'*newA(:,p))*newA(:,p) ;
        end
            
        tempdist(i) = norm(diffA,'fro')^2/norm(A,'fro')^2 ;
    end
    tmsfe = tmsfe + min(tempdist) ;
end
        
    