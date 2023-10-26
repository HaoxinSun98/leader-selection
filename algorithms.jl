include("graph.jl")
include("graphw.jl")
include("edgecore.jl")
include("baseline.jl")

using LinearAlgebra
using Random
using Statistics
using Laplacians
using SparseArrays
using DataFrames
using Combinatorics, CSV


function Naive(G,s,k)
    tt = zeros(k+1);
    t1 = time()
    H=[];
    anslist = zeros(k+1);
    L = lap(G);
    for i = 1:G.n
        L[i, i] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    t2 = time()
    tt[1] = t2-t1
    for t = 1:k
        deltaHi = zeros(G.n,1);
        for i = 1:G.n
            if  !(i in H)
                xx = (onee*L[:,i]*L[i,:]'*s)./L[i,i];
                deltaHi[i] = xx[1];
            end
        end
        x = findmax(deltaHi);
        xx = x[2][1];
        push!(H,xx);
        L = L - L[:,xx]*L[xx,:]'./L[xx,xx];
        anslist[t+1] = x[1];
        t2 = time();
        tt[t+1] = t2-t1
    end
    b2 = zeros(k);
    for j = 1:k
        b2[j] = anslist[1] - sum(anslist[2:(j+1)]);
    end
    return [b2,tt[2:end]]
end

function Naive_w(G,s,k)
    tt = zeros(k+1);
    t1 = time()
    H=[];
    anslist = zeros(k+1);
    L = lap_w(G);
    for i = 1:G.n
        L[i, i] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    t2 = time()
    tt[1] = t2-t1
    for t = 1:k
        deltaHi = zeros(G.n,1);
        for i = 1:G.n
            if  !(i in H)
                xx = (onee*L[:,i]*L[i,:]'*s)./L[i,i];
                deltaHi[i] = xx[1];
            end
        end
        x = findmax(deltaHi);
        xx = x[2][1];
        push!(H,xx);
        L = L - L[:,xx]*L[xx,:]'./L[xx,xx];
        anslist[t+1] = x[1];
        t2 = time();
        tt[t+1] = t2-t1
    end
    b2 = zeros(k);
    for j = 1:k
        b2[j] = anslist[1] - sum(anslist[2:(j+1)]);
    end
    return [b2,tt[2:end]]
end


function Fast_w(G,s,tau,k)
    tt = zeros(k)
    t1 = time();
    d = [sum(G.w[j]) for j = 1:G.n];
    ansvalue = zeros(k);
    DeltaH1 = zeros(G.n);
    DeltaH2 = zeros(G.n);
    q = zeros(G.n);
    U=[];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);
    rootinU = [false for j = 1:G.n];
    deltaexpressed = zeros(G.n);
    sumS = zeros(G.n);
    N = zeros(G.n);
    for t = 1:k
        DeltaH1.=0;
        DeltaH2.=0;
        q.=0;
        deltaexpressed .=0;
        UU = union(1:G.n);
        setdiff!(UU,U);
        t1  = time();
        for p = 1:tau
            InForest .= false;
            rootinU .= false;
            Next .= -1;
            Rootindex .= 0;
            sumS .= 0;
            N .=0;
            for i in U
                InForest[i] = true;
                rootinU[i] = true;
                Rootindex[i] =i;
            end                 # U里面的点已经是根了
            for i = 1:G.n
                u = i;
                while !InForest[u];
                    seed = rand();
                    if seed <= 1 / (1 + d[u])
                        InForest[u] = true;
                        Next[u] = -1;
                        Rootindex[u] = u;
                        N[u] += 1;
                        sumS[u] += s[u];
                    else
                        seed = rand();
                        sumw = 0;
                        k = 0;
                        while sumw<seed
                            k += 1;
                            sumw += G.w[u][k]/d[u];
                        end
                        Next[u] = G.nbr[u][k];
                        u = Next[u];
                    end
                end
                rootnow = Rootindex[u];
                rootinUnow = rootinU[u];
                u = i
                while !InForest[u]
                    InForest[u] = true;
                    Rootindex[u] = rootnow;
                    rootinU[u] = rootinUnow;
                    if !rootinUnow
                        N[rootnow] += 1;
                        sumS[rootnow] += s[u];
                    end
                    u = Next[u];
                end
            end
            for i = 1:G.n
                if !rootinU[i]
                    DeltaH1[i] += 1;
                    DeltaH2[i] += sumS[Rootindex[i]]/N[Rootindex[i]];
                    q[i] += 1/N[Rootindex[i]];
                end
            end
        end
        deltaexpressed = DeltaH1.*DeltaH2 ./q;
        for i =1:G.n
             if isnan(deltaexpressed[i])
                 deltaexpressed[i] = -Inf
             end
         end
        x = findmax(deltaexpressed);
        push!(U,x[2])
        ansvalue[t]=x[1]/tau;
        t2 = time();
        tt[t] = t2-t1;
    end
    return [U,tt]
end



function Fast(G,s,tau,k)
    tt = zeros(k)
    t1 = time();
    d = [length(G.nbr[j]) for j = 1:G.n];
    ansvalue = zeros(k);
    DeltaH1 = zeros(G.n);
    DeltaH2 = zeros(G.n);
    q = zeros(G.n);
    U=[];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);
    rootinU = [false for j = 1:G.n];
    deltaexpressed = zeros(G.n);
    sumS = zeros(G.n);
    N = zeros(G.n);
    for t = 1:k
        DeltaH1.=0;
        DeltaH2.=0;
        q.=0;
        deltaexpressed .=0;
        UU = union(1:G.n);
        setdiff!(UU,U);
        for p = 1:tau
            InForest .= false;
            rootinU .= false;
            Next .= -1;
            Rootindex .= 0;
            sumS .= 0;
            N .=0;
            for i in U
                InForest[i] = true;
                rootinU[i] = true;
                Rootindex[i] =i;
            end                 # U里面的点已经是根了
            for i = 1:G.n
                u = i;
                while !InForest[u];
                    seed = rand();
                    if seed <= 1 / (1 + d[u])
                        InForest[u] = true;
                        Next[u] = -1;
                        Rootindex[u] = u;
                        N[u] += 1;
                        sumS[u] += s[u];
                    else
                        k = floor(Int, seed * (1 + d[u]));
                        Next[u] = G.nbr[u][k];
                        u = Next[u];
                    end
                end
                rootnow = Rootindex[u];
                rootinUnow = rootinU[u];
                u = i
                while !InForest[u]
                    InForest[u] = true;
                    Rootindex[u] = rootnow;
                    rootinU[u] = rootinUnow;
                    if !rootinUnow
                        N[rootnow] += 1;
                        sumS[rootnow] += s[u];
                    end
                    u = Next[u];
                end
            end
            for i = 1:G.n
                if !rootinU[i]
                    DeltaH1[i] += 1;
                    DeltaH2[i] += sumS[Rootindex[i]]/N[Rootindex[i]];
                    q[i] += 1/N[Rootindex[i]];
                end
            end
        end

        deltaexpressed = DeltaH1.*DeltaH2 ./q;
        for i =1:G.n
             if isnan(deltaexpressed[i])
                 deltaexpressed[i] = -Inf
             end
         end
        x = findmax(deltaexpressed);
        push!(U,x[2])
        ansvalue[t]=x[1]/tau;
        t2 = time();
        tt[t] = t2-t1;
    end
    return [U,tt]
end




function iteronce(old,s,A,invid)
    return invid.*s+A*old
end


function Greedy_w(G,s,k,l)
    tt = zeros(k);
    t1 = time()
    d = [sum(G.w[j]) for j = 1:G.n];
    invid = zeros(G.n);
    for i = 1:G.n
        invid[i] = 1/(1+d[i]);
    end
    A = spzeros(G.n,G.n);
    for i  = 1:G.n
        for j = 1:length(G.w[i])
            A[i,G.nbr[i][j]] = invid[i]*G.w[i][j];
        end
    end
    H=[];
    anslist = zeros(k);
    for t = 1:k
        HH = union(1:G.n);
        setdiff!(HH,H);
        sumzwithi = zeros(G.n);
        z0 = zeros(G.n,1);
        for i in HH
            #z0 = deepcopy(s);
            z0 .= s;
            z0[i] = 0;
            z0[H] .= 0;
            z = iteronce(z0,z0,A,invid);
            z[i] = 0;
            z[H] .= 0;
            for j = 1:l-1
                z = iteronce(z,z0,A,invid)
                z[i] = 0;
                z[H] .= 0;
            end
            sumzwithi[i] = sum(z);
        end
        for i = 1:G.n
            if sumzwithi[i] == 0
                sumzwithi[i] = Inf;
            end
        end
        xx = argmin(sumzwithi)
        #x = findmin(sumzwithi);
        #xx = x[2];
        push!(H,xx);
        anslist[t] = sumzwithi[xx];
        t2 = time();
        tt[t] = t2-t1
    end
    anslist = zeros(k+1);
    L = lap_w(G);
    for i = 1:G.n
        L[i, i] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    LL = deepcopy(L);
    for t = 1:k
        ii =H[t];
        xx = (onee*LL[:,ii]*LL[ii,:]'*s)./LL[ii,ii];
        LL = LL - LL[:,ii]*LL[ii,:]'./LL[ii,ii];
        anslist[t+1] = anslist[t]-xx[1];
    end
    return [anslist[2:k+1],tt]
end



function Greedy(G,s,k,l)
    tt = zeros(k);
    t1 = time()
    d = [length(G.nbr[j]) for j = 1:G.n];
    invid = zeros(G.n);
    for i = 1:G.n
        invid[i] = 1/(1+d[i]);
    end
    uu=zeros(2*G.m);
	vv=zeros(2*G.m);
	ww=zeros(2*G.m);
	uu[1:G.m]=G.u;uu[G.m+1:2*G.m]=G.v;
	vv[1:G.m]=G.v;vv[G.m+1:2*G.m]=G.u;
	for i =1:G.m
        ww[i] = invid[G.u[i]];
    end
    for i =(G.m+1):2*G.m
        ww[i] = invid[G.v[i-G.m]];
    end
    A = sparse(uu,vv,ww);
    H=[];
    anslist = zeros(k);
    for t = 1:k
        HH = union(1:G.n);
        setdiff!(HH,H);
        sumzwithi = zeros(G.n);
        z0 = zeros(G.n,1);
        for i in HH
            #z0 = deepcopy(s);
            z0 .= s;
            z0[i] = 0;
            z0[H] .= 0;
            z = iteronce(z0,z0,A,invid);
            z[i] = 0;
            z[H] .= 0;
            for j = 1:l-1
                z = iteronce(z,z0,A,invid)
                z[i] = 0;
                z[H] .= 0;
            end
            sumzwithi[i] = sum(z);
        end
        for i = 1:G.n
            if sumzwithi[i] == 0
                sumzwithi[i] = Inf;
            end
        end
        xx = argmin(sumzwithi)
        #x = findmin(sumzwithi);
        #xx = x[2];
        push!(H,xx);
        anslist[t] = sumzwithi[xx];
        t2 = time();
        tt[t] = t2-t1
    end
    anslist = zeros(k+1);
    L = lap(G);
    for ii = 1:G.n
        L[ii, ii] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    LL = deepcopy(L);
    for t = 1:k
        ii =H[t];
        xx = (onee*LL[:,ii]*LL[ii,:]'*s)./LL[ii,ii];
        LL = LL - LL[:,ii]*LL[ii,:]'./LL[ii,ii];
        anslist[t+1] = anslist[t]-xx[1];
    end
    
    return [anslist[2:k+1],tt]
end



function Lapsolver(G,s,k)
    tt = zeros(k+1);
    t1 = time()
    H=[];
    anslist = zeros(k+1);
    L = lapsp(G);
    for i = 1:G.n 
        L[i,i]+=1;
    end
    f = approxchol_sddm(L)
    anslist[1] = sum(f(s));
    onee = ones(1,G.n);
    F = union(1:G.n)
    t2 = time()
    tt[1] = t2-t1
    for t = 1:k
        deltaHi = zeros(G.n,1);
        for i = 1:G.n
            if  !(i in H)
                aa = zeros(G.n);
                aa[i] = 1;
                aa = aa[F]
                kkk = f(aa)
                xx = (onee[F]'*kkk)*(aa'*f(s[F]))/(aa'*kkk)
                deltaHi[i] = xx[1];
            end
        end
        x = findmax(deltaHi);
        xx = x[2][1];
        push!(H,xx);
        F = union(1:G.n)
        setdiff!(F,H)
        f = approxchol_sddm( L[F,F]);
        anslist[t+1] = x[1];
        t2 = time();
        tt[t+1] = t2-t1
    end
    anslist = zeros(k+1);
    L = lap(G);
    for ii = 1:G.n
        L[ii, ii] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    LL = deepcopy(L);
    for t = 1:k
        ii =H[t];
        xx = (onee*LL[:,ii]*LL[ii,:]'*s)./LL[ii,ii];
        LL = LL - LL[:,ii]*LL[ii,:]'./LL[ii,ii];
        anslist[t+1] = anslist[t]-xx[1];
    end
    return [ anslist[2:k+1],tt[2:end]]
end


function Lapsolver_w(G,s,k)
    tt = zeros(k+1);
    t1 = time()
    H=[];
    anslist = zeros(k+1);
    L = lap_w(G);
    L = sparse(L)
    for i = 1:G.n 
        L[i,i]+=1;
    end
    f = approxchol_sddm(L)
    anslist[1] = sum(f(s));
    onee = ones(1,G.n);
    F = union(1:G.n)
    t2 = time()
    tt[1] = t2-t1
    for t = 1:k
        deltaHi = zeros(G.n,1);
        for i = 1:G.n
            if  !(i in H)
                aa = zeros(G.n);
                aa[i] = 1;
                aa = aa[F]
                kkk = f(aa)
                xx = (onee[F]'*kkk)*(aa'*f(s[F]))/(aa'*kkk)
                deltaHi[i] = xx[1];
            end
        end
        x = findmax(deltaHi);
        xx = x[2][1];
        push!(H,xx);
        F = union(1:G.n)
        setdiff!(F,H)
        f = approxchol_sddm( L[F,F]);
        anslist[t+1] = x[1];
        t2 = time();
        tt[t+1] = t2-t1
    end
    anslist = zeros(k+1);
    L = lap_w(G);
    for ii = 1:G.n
        L[ii, ii] += 1;
    end
    L = inv(L);
    anslist[1] = sum(L*s);
    onee = ones(1,G.n);
    LL = deepcopy(L);
    for t = 1:k
        ii =H[t];
        xx = (onee*LL[:,ii]*LL[ii,:]'*s)./LL[ii,ii];
        LL = LL - LL[:,ii]*LL[ii,:]'./LL[ii,ii];
        anslist[t+1] = anslist[t]-xx[1];
    end
    return [ anslist[2:k+1],tt[2:end]]
end



