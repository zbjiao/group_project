insurance = readtable('insurance.csv');
data = insurance;
%Transfer categorical value to numerical value
X = data(:,1:6);
Y = data(:,7);
[GN, ~, G] = unique(X(:,2));
X.sex = G-1;
X.smoker = strcmp(X.smoker, 'yes');
%check the variables names
data.Properties.VariableNames;
figure(1)
histogram(data.age,'BinWidth',5);
xlabel('Age');
ylabel('Frequency');
figure(2)
C = categorical(X.sex,[0 1],{'Female','Male'});
histogram(C,'BarWidth',0.5);
xlabel('Sex');
ylabel('Frequency');
figure(3)
histogram(data.bmi);
xlabel('bmi');
ylabel('Frequency');
figure(4)
histogram(data.children);
xlabel('children');
ylabel('Frequency');
figure(5)
C0 = categorical(X.smoker,[0 1],{'No','Yes'});
histogram(C0, 'BarWidth',0.5);
xlabel('smoker');
ylabel('Frequency');
figure(6)
C1 = categorical(data.region);
histogram(C1, 'BarWidth',0.5);
xlabel('Region');
ylabel('Frequency');
figure(7)
histogram(data.charges,'BinWidth',5000);
xlabel('charges');
ylabel('Frequency');
%The response has heavy tail distribution, to fix this, we take the log of
%response, now
Y.charges = log(Y.charges);
figure(8)
histogram(Y.charges,'BinWidth',0.5);
xlabel('ln-charges');
ylabel('Frequency');
% Now it is normal
% Now, to regress the data, we should convert region to dummy variables
X.southwest = ( strcmp(X.region ,'southwest'));
X.northwest = ( strcmp(X.region ,'northwest'));
X.southeast = ( strcmp(X.region ,'southeast'));
X = removevars(X,{'region'});


x = X{:,:};
[m,p] = size(x);
y = Y{:,:};
n = length(y);
mdl = fitlm(x,y);
mdl.ModelCriterion;
%Plot residual
figure(9)
plotResiduals(mdl);
%Normal probability plot
figure(10)
plotResiduals(mdl,'probability');
figure(11)
plotResiduals(mdl,'fitted');



%Use an interactive model to modify the orginal model: 
x1 = X;
x1.smokerbmi = x1.smoker.*x1.bmi;
x11 = x1{:,:};
[m1,p1] = size(x11);
mdl2 = fitlm(x11,y);
mdl2.ModelCriterion;
%AIC and BIC are both lower, so we retain smoker and bmi interaction
%predictor

%Now, we add another interactive predictor smoker:age
x2 = X;
x2.smokerbmi = x2.smoker.*x2.bmi;
x2.smokerage = x2.smoker.*x2.age;
x12 = x2{:,:};
[m2,p2] = size(x12);
mdl3 = fitlm(x12,y);
mdl3.ModelCriterion;
% BIC decrease, so we retain this predictor

%Add bmi^1.5
x3 = x2;
x3.bmi = x3.bmi+x3.bmi.^1.5;
x13 = x3{:,:};
[m3,p3] = size(x13);
mdl4 = fitlm(x13,y);
mdl4.ModelCriterion;

%Add age^1.5
x4 = x3;
x4.age = x4.age+x4.age.^1.5;
x14 = x4{:,:};
mdl5 = fitlm(x14,y);
mdl5.ModelCriterion;
% Now, the fitted value and response are almost independent.
figure(12)
plotResiduals(mdl5,'fitted');

%Model selection
%What if we regress the origin response without taking log?
x_1 = x4;
x_11 = x_1{:,:};
y_1 = data.charges;
mdl6 = fitlm(x_11,y_1);
mdl6.ModelCriterion;



% It turns out that region is not a very significant predictor, so we only
% keep southwest which has the lowest P-value

%Refit the model
x_2 = x2;
x_2.age15 = x_2.age.^1.5;
x_2.bmi15 = x_2.bmi.^1.5;
x_2 = removevars(x_2,{'southwest','southeast'});
x_12 = x_2{:,:};
y_1 = data.charges;
mdl7 = fitlm(x_12,y_1);
mdl7.ModelCriterion;
%We see a reduction in AIC and BIC
%Smokerage has high P-value, so we drop it;
x_3 = x_2;
x_3 = removevars(x_3,{'smokerage'});
x_13 = x_3{:,:};
y_1 = data.charges;
mdl8 = fitlm(x_13,y_1);
mdl8.ModelCriterion;


