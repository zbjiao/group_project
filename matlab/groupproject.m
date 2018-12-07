%insurance = readtable('insurance.csv');
data = insurance;
X = data(:,1:6);
Y = data(:,7);
[GN, ~, G] = unique(X(:,2));
X.sex = G-1;
[CN, ~, C] = unique(X(:,5));
X.smoker = C - 1;

%Solve the least square problem (without using region as predictor):
x = X{:,1:5};
[m,p] = size(x);
x = [ones(m,1),x];
y = Y{:,:};
n = length(y);
beta_hat = inv(transpose(x)*x) * transpose(x) * y;
RSS = transpose(y - x*beta)*(y - x*beta);
TSS = transpose(y - mean(y))*(y - mean(y));
sigma_hat_sq = RSS/(n-p-1);
sigma_hat = sqrt(sigma_hat_sq);
%Calculate p_value, since n is large, the t - test result is close to
%normal result.
P_value = 2 * (normpdf(abs(beta_hat),0,sigma_hat));
%Using CI of 95%, age and smoker are identified as significant predictors. 
AIC = n * log(RSS/n) +2*(p+1);
BIC = n * log(RSS/n) +(p+1)*log(n);
R_square = RSS/TSS;
Adjusted_R_square = 1 - (n-1)/(n-p-1)*(1-R_square);
%now we complete computing all required value for one model. We repeat this
%process for other models. 
