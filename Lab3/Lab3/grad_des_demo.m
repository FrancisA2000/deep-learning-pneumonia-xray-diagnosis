% z(w1,w2) = 2*(w1-1)^2 + w2^2
% grad = [4*(w1-1),2*w2]^T
% Minimum of z(w1,w2) at w1_opt=1, w2_opt=0
clear 
close all
range = -3:0.25:4;
[W1,W2] = meshgrid(range);
Z = (2*(W1-1).^2+W2.^2);
contour3(W1,W2,Z,30)
xlabel('w_1')
ylabel('w_2')
zlabel('z(w_1,w_2)')

w1=-3; w2=3;
lambda = 1e-2; %5e-1

for t=1:200
    g_w1=4*(w1(t)-1);
    g_w2=2*w2(t);
    z(t)=2*(w1(t)-1)^2+w2(t)^2;
    disp([t,w1(t),w2(t),z(t)])
    w1(t+1)=w1(t)-lambda*g_w1;
    w2(t+1)=w2(t)-lambda*g_w2;
    hold on
    plot3(w1(1:t),w2(1:t),z(1:t),'rx')
    pause(0.25)
end