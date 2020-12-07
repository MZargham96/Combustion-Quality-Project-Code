clc; close all;

%this code applies a fully connected Neutral Network over the pressure data 
%Pressure data has been standard normalized and its dimensions have been
%reduced using Linear PCA

%Pressure Data imported from excel
T1 = readtable('pressure_data_raw2', 'Range', 'A1:GT154');
T2A = table2array(T1);
%%
%sort data into x and y
x = T2A(:,2:end)';
y = T2A(:,1)';
y(y==0) = -1;
P = size(x,2);

%sorting data into testing and training sets
one = find(y==1);
minusone = find(y==-1);
onebythree = round(2/3*length(one));

% training data set and standard normalization
ytrain = [y(one(1,1:onebythree)) y(minusone(1,1:onebythree))];
xtrain = [x(:,one(1,1:onebythree)) x(:,minusone(1,1:onebythree))];

Means = mean(xtrain,2);
S = std(xtrain,1,2);
Xtrain = (xtrain - Means)./S;

%testing set and standard normalization
ytest = [y(one(1,onebythree+1:end)) y(minusone(1,onebythree+1:end))];
xtest = [x(:,one(1,onebythree+1:end)) x(:,minusone(1,onebythree+1:end))];

Means = mean(xtest,2);
S = std(xtest,1,2);
Xtest = (xtest - Means)./S;
%%
% find PCA
Cov = (1/P)*Xtrain*Xtrain'; %Covariance Matrix
[V,D] = eig(Cov);
[D_sorted,id] = sort(diag(D),'descend'); %sorting eigenvalues
A = D_sorted(D_sorted>0.0005); 

% choose spanning set
Ctrain = [];
for i = 1:length(A)
    Ctrain = horzcat(Ctrain, V(:,id(i)));
end

figure(1)
plot(id,D_sorted)
title('Principal Component Analysis of Raw Pressure Data'); 
xlabel('Dimensions'); ylabel('Eigenvalues');
xlim([0 201]);

% project data  onto new spanning set
wptrain = linsolve(Ctrain'*Ctrain,Ctrain'*Xtrain);
wptest = linsolve(Ctrain'*Ctrain,Ctrain'*Xtest);
%%
%Defining the Neural Network
%number of layers
L=3;
%number of units
U=3;
%number of features 
N=size(wptrain,1);

%Weights Intialization
Wn = (N+1)*U+(L-1)*(U+1)*U+U+1;
w0 = randn(Wn,1);

%training the classifier
f = @(w)softmax(w,wptrain,ytrain,L,U);
[W,fW] = gradient_descent(f,w0,0.001,12000);
W_opt = W(:,end);

%figure showing cost optimization w.r.t iteration 
cost = softmax(W,wptrain,ytrain,L,U);
figure(2)
plot(cost)
xlabel('Iteration'); ylabel('Cost/ Error');
title('Cost minimimization for Pressure Data Raw')
%%
%Defining Threshold for analysis and accuracy measurement
atrain = model(W_opt,wptrain,L,U);
atrain(atrain>0) = 1;
atrain(atrain<0) = -1;
        
Ctrain = confusionmat(atrain,ytrain)
Acc = 0.5*(Ctrain(1,1)/sum(Ctrain(:,1)))...
    + 0.5*(Ctrain(2,2)/sum(Ctrain(:,2)));
        
Abalancedtraining = Acc*100

atest = model(W_opt,wptest,L,U);
atest(atest>0) = 1;
atest(atest<0) = -1;
        
Ctest = confusionmat(atest,ytest)
Acc = 0.5*(Ctest(1,1)/sum(Ctest(:,1)))...
    + 0.5*(Ctest(2,2)/sum(Ctest(:,2)));
        
Abalancedtesting = Acc*100

%%
function a = model(w,wptrain,L,U) %non-linear model
N = size(wptrain,1);
W{1} = reshape(w(1:(N+1)*U),N+1,U);
for j = 1:L-1
    W{j+1} = reshape(w((N+1)*U+(j-1)*(U+1)*U+1 :...
        (N+1)*U+j*(U+1)*U),U+1,U);
end
W{L+1} = w((N+1)*U+(L-1)*(U+1)*U+1 : end);

xbar = [ones(1,size(wptrain,2));wptrain];
a = tanh(xbar'*W{1})';

for i = 2:L+1
    abar = [ones(1,size(a,2));a];
    a = tanh(abar'*W{i})';
end

end

function cost = softmax(w,wptrain,ytrain,L,U)
P = length(ytrain);
for i = 1 : size(w,2)
    wi = w(:,i);
    model_i = model(wi,wptrain,L,U);
    cost(i) = (1/P) * ...
        sum( ...
        log( 1 + exp(-ytrain.*model_i) ));
end
end

function [W,fW] = gradient_descent(f,w0,alpha,n_iter)

k = 1;
W = w0;
fW = f(w0);

while k < n_iter
    grad = approx_grad(f,W(:,k),.000001);
    W(:,k+1) = W(:,k) - alpha*(grad')/(norm(grad)+0.0001);
    fW(k+1) = f(W(:,k+1));
    k = k+1;
end

end

function grad = approx_grad(f,w0,delta)
N = length(w0);
dw = delta*eye(N);
grad = ( f(w0+dw) - f(w0) )/delta;
end
