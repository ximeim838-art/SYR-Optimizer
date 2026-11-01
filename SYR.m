function [best, Convergence_curve] = SYR(N, MaxFEs, lb, ub, dim, fobj)
pm = 0.005;
X = lb + rand(N, dim) .* (ub - lb);
X = max(min(X, ub), lb);
Fit = zeros(N,1);
for i = 1:N
    Fit(i) = fobj(X(i,:));
end
FEs = N;
[Fbest, idx] = min(Fit);
best = X(idx,:);
Convergence_curve = [];
Time = 1;
stagnant_count = zeros(N,dim);
stagnant_limit = 33;
while FEs < MaxFEs
    progress = FEs / MaxFEs;
    [~, rank_idx] = sort(Fit,'ascend');
    p = 1 + progress;
    scent_rank = exp(-((1:N)'/N).^p);
    scent_strength = zeros(N,1);
    scent_strength(rank_idx) = scent_rank;
    scent_weight = scent_strength / sum(scent_strength);
    scent_center = scent_weight' * X;
    for i = 1:N
        if FEs >= MaxFEs, break; end
        xi = X(i,:);
        if progress < rand()
            j = randi(N);
            dir = (X(j,:) - xi) + progress * (scent_center - xi);
            if norm(dir)>0, dir = dir/norm(dir); end
            d = norm(xi - scent_center);
            R = (1-progress)^2;
            r1 = 2*rand(1,dim)-1;
            if rand < pm
                  u = (scent_center - xi);
                if norm(u) > 0
                    u = u / norm(u);
                end
                D = 0.5*(1-progress)*tanh(norm(xi-best)/sqrt(dim));
                X_new = xi + R*d.*u.*D;
            else
                X_new = xi + d*R.*dir.*r1;
            end
        else
            dir = zeros(1,dim);
            for t = 1:N
                d = norm(X(t,:) - xi);
                if d == 0, continue; end
                w = scent_strength(t) * exp(-d); 
                dir = dir + w * (X(t,:) - xi);
            end
            if norm(dir)>0, dir = dir/norm(dir); end
            r = randperm(N,3);
            while any(r==i)
                r = randperm(N,3);
            end
            diff_vec = X(r(1),:) - X(r(2),:);
            alpha = 0.5 * norm(best - xi) * (1 - progress);
            X_new = xi;           
            stuck_dim = stagnant_count(i,:) > stagnant_limit;
            if ~any(stuck_dim)
                k = max(1,ceil(rand*dim));
                stuck_dim(randperm(dim,k)) = true;
            end
            X_new(stuck_dim) =  best(stuck_dim)   + alpha * dir(stuck_dim) + (1-alpha) * diff_vec(stuck_dim).*cos(pi*(rand)) ;
        end
        X_new = max(min(X_new, ub), lb);
        F_new = fobj(X_new);
        FEs = FEs + 1;
        if F_new < Fit(i)
            improve_mask = abs(X_new - X(i,:)) > 1e-12;
            X(i,:) = X_new;
            Fit(i) = F_new;
            stagnant_count(i,improve_mask) = 0;
            stagnant_count(i,~improve_mask) = stagnant_count(i,~improve_mask) + 1;
            if F_new < Fbest
                Fbest = F_new;
                best = X_new;
            end
        else
            stagnant_count(i,:) = stagnant_count(i,:) + 1;
        end
    end
    Convergence_curve(Time) = Fbest;
    Time = Time + 1;
end
end
