% this is problem 3
% objective function x2+10^-5(x2-x1)^2
% constraint x2>=0
% using line search sqp algorithm
%% initialization 
clc;
clear;
disp('Using line search sqp algorithm to solve problem');
tic; % start to time
n = 0.1; %select n from 0 to 0.5
tau = 0.5; %select tau from 0 to 1
lambdanow = 0; %initialize lambda to 0
x1 = 10; %starting x1
x2 = 1; %starting x2
c_0 = c(x1, x2); %starting constraintt with x1,x2
A_0 = c1(x1, x2); %starting A with x1,x2
 
% this is my lagrangian function, it is constant matrix calculated manually.
% because my hessian of objective function is constant, and my hessian of
% constraint is equal to 0. Therefore, whatever lambda is, it always equal
% to hessian of my objective function.
lagran = 2*10^(-5)*[1,-1;-1,1]; 
 
ct = 1; % count number of iteration
 
while c_0 > 10^-9 || c_0 < -10^-9
%     to start 18.11
    f_0 = f(x1, x2); % get current f(x) value
    f_1 = f1(x1, x2); % get current gradient of f(x) value
    
    c_0 = c(x1, x2); % get current c(x) value
    c_1 = c1(x1, x2); % get current gradient c(x) value
    [pk,qfval,qexitflag,qoutput,headlambda] = quadprog(lagran,f_1,-c_1,c_0); %solve subproblem in 18.11
    lambdap = headlambda.ineqlin - lambdanow; % get lamdap value
%     start 18.36 with o = 1;
    uk = (f_1*pk+0.5*pk'*lagran*pk)/((1-0.5)*abs(x2)); % following 18.36 to get u value
    ak = 1; %set step length to 1
%     start next while loop for merit function
    takex1 = x1 + ak*pk(1); %used for first merit function
    takex2 = x2 + ak*pk(2); %used for first merit function
    r1 = f(takex1, takex2) + uk*abs(x2); %merit function1
    r2 = f(x1, x2) + uk*abs(x2); %merit function2
    r3 = n*ak*(f1(x1, x2)*pk - uk*abs(x2)); %merit function3
    % below is while loop to get suitable step length value
    while r1 > r2 + r3
        ak = 0.3*ak; % set new step length value
        r1 = f(takex1, takex2) + uk*abs(x2); %merit function update
        r2 = f(x1, x2) + uk*abs(x2); %merit function update
        r3 = n*ak*(f1(x1, x2)*pk - uk*abs(x2)); %merit function update
    end
    x1 = x1 + ak*pk(1); % update x1 value by direction and step length
    x2 = x2 + ak*pk(2); % update x2 value by direction and step length
    disp(strcat('iteration: ', num2str(ct))); % display iteration number
    disp(strcat('x1 value: ', num2str(x1))); %display current x1 value
    disp(strcat('x2 value: ', num2str(x2))); % display current x2 value
    f_0 = f(x1, x2); % get current f value
    disp(strcat('f value: ', num2str(f_0))); % display current f value
    f_1 = f1(x1, x2); % update f gradient value
    c_0 = c(x1, x2); % update current constraint value
    A_0 = c1(x1, x2); % update gradient of constraint value
    ct = ct + 1; % iteration update
    if (ct > 100)
        disp('converge fail');
        break
    end
end
if ct <= 100
    disp('converge successful');
end
disp(strcat('total number of iteration: ', num2str(1))); % display total iteration
sol = f(x1, x2); % final solution
prot = toc; % stop timing
disp(strcat('Run time of line search sqp: ', num2str(prot))); % display running time
disp(strcat('optimal x1 value: ', num2str(x1))); % display optimal x1 value
disp(strcat('optimal x2 value: ', num2str(x2))); % display optimal x2 value
disp(strcat('optimal f value: ', num2str(f_0))); % display optimal f value
%% commercial solver
tic;
fun = @(x) x(2) + 10^(-5) * (x(2) - x(1))^2;
x0 = [10, 1];
A = [0, -1];
b = 0;
[x, fval] = fmincon(fun, x0, A, b);
prot = toc;
disp(strcat('Run time of commercial solver: ', num2str(prot)));
disp(strcat('optimal x value for commercial solver: ', num2str(x)));
disp(strcat('optimal f value for commercial solver: ', num2str(fval)));
