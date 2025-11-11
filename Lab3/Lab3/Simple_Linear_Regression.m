clear all
close
load Advertising.mat
x=Advertising(:,2:2);
X=[ones(200,1), x];
Y=Advertising(:,5);
w_opt=pinv(X)*Y;
plot(x,Y,'x',x,X*w_opt)
legend('Data','Regression')
xlabel('TV Advertising Budget [K$]')
ylabel('Sales [M$]')
title(['Linear Regression with 1 Variable: MSE=',num2str(mse(X*w_opt-Y))])