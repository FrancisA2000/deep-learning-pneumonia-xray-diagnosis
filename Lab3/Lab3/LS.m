close all
clear 
load Advertising.mat
X=[ones(200,1), Advertising(:,2:4)];
Y=Advertising(:,5);
w_opt=pinv(X)*Y;
w=zeros(4,1);
lr=2e-7;
iters=1500000;
E=zeros(iters/1000,1);
W=zeros(4,iters/1000);
for i=1:iters
    if mod(i,1000)==0
        i
        E(i/1000)=mse(X*w-Y);
        W(:,i/1000)=w;
    end
    G=(X'*X*w-X'*Y);
    w=w-lr*G;
end
figure; plot(W'); grid on; xlabel('Iterations [/1000]'); ylabel('W')
legend('w_0','w_1','w_2','w_3')
figure; plot(E); grid on; xlabel('Iterations [/1000]'); ylabel('MSE')
w