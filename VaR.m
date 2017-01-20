function[parametric,parametric2,historical,monte] = VaR(price,port,percent,day,window,options)
%compute all types of VaR given historical data; exponantional weighting
%Assume GBM
%price is the historical price of stocks, which is x*y
%port is the shares of different equities in portfolio, include normal stock, call options and put options, which is 1*y
%percent is the percent of VaR, which is a number between 0 and 1
%day is the days to calculate VaR, window is the year window
%options data only includes the strike price of each option corresponding to each stock in price data


%Use Black-Scholes to simulate option price. Assume rate, maturity time and volatility to be constant
[x,y] = size(price);
Call = zeros(x,y);
Put = zeros(x,y);
for i = 1:x
    for j = 1:y
        Strike = options(j);
        Rate = 0.01;
        Time = 1;
        Volatility = 0.2;
        [Call(i,j), Put(i,j)] = blsprice(price(i,j), Strike, Rate, Time, Volatility);
    end
end

%Combine stock and option price together
price = [price,Call,Put];
[m,n] = size(price);
%calculate portfolio value
value = zeros(m,1);
for i = 1:m
    for j = 1:n
        value(i) = value(i) + price(i,j)*port(j);
    end
end

%calculate log return
lreturn = zeros(m-1,1);
for i = 1:m-1
    lreturn(i)= log(value(i)/value(i+1));
end
%square log return
slreturn = lreturn.^2;

%calculate lambda with a weight of 20%
lambda = nthroot(0.2,252*window);
%construct lambda array
Lambda = zeros(m,1);
for i = 1:m
   Lambda(i) = lambda^i;
end
%exponantial weighting lambda
weighted = zeros(m,1);
for i = 1:m
    weighted(i) = Lambda(i)/sum(Lambda(1:(window*252)));
end

%compute sigma and mu
a = m-window*252;
sigma = zeros(a,1);
mu = zeros(a,1);

for i = 1:a
    b = sum(slreturn(i:(i+252*window-1)).*weighted(1:window*252));
    c = sum(lreturn(i:(i+252*window-1)).*weighted(1:window*252));
    sigma(i) = sqrt(b - c^2)*sqrt(252*window);
end

for i = 1:a
    d = sum(lreturn(i:(i+252*window-1)).*weighted(1:window*252));
    mu(i) = d*252 + sigma(i)^2/2;
end

%compute the initial capital (portfolio value of the first day in trading history)
v0 = sum(price(m,:).*port);

%compute exponential weighting parametric VaR
parametric = zeros(a,1);
for i = 1:a
    e = norminv(1-percent,0,1);
    parametric(i) = v0*(1-exp(sigma(i)*sqrt(day/252)*e+(mu(i)-sigma(i)^2/2)*day/252));
end


for i = 1:a
    e = norminv(1-percent,0,1);
    parametric(i) = v0*(1-exp(sigma(i)*sqrt(day/252)*e+(mu(i)-sigma(i)^2/2)*day/252));
end

%compute windowed parametric VaR
parametric2 = zeros(a,1);
sigma2 = zeros(a,1);
mu2 = zeros(a,1);
for i = 1:a
    sigma2(i) = std(lreturn(i:i + 252*window - 1))*sqrt(252);
end
for i = 1:a
    mu2(i) = mean(lreturn(i:i+window*252-1))*252 + sigma(i)^2/2;
end
for i = 1:a
    e = norminv(1-percent,0,1);
    parametric2(i) = v0*(1-exp(sigma2(i)*sqrt(day/252)*e+(mu2(i)-sigma2(i)^2/2)*day/252));
end


%compute historical VaR, relative changes
historical = zeros(m-1,1);

for i = 1:m-1
    %construct historical samples
    hsample = zeros(m-i,1);
    for j = 1:m-i
        hsample(j) = value(i)*lreturn(j);
    end
    %compute relative losses
    loss = zeros(m-i,1);
    for j = 1:m-i
        loss(j) = value(j) - hsample(j);
    end
    historical(i) = prctile(loss,100*percent);
end


%compute Monte Carlo VaR using GBM parameters computed above
w=zeros(1,10000);
P=zeros(a,10000);
monte=zeros(a,1);

for i=1:a
    w(1,:) = mvnrnd(0,1,10000);
    P(i,:) = v0*exp((mu(i)-sigma(i)^2/2)*(day/252)+sigma(i)*w(1,:)*sqrt(day/252));
    monte(i) = quantile(10000-P(i,:),percent);
    
end

figure
subplot(4,1,1)       % add first plot in 4 x 1 grid
plot(parametric)
title('exponentially weighted Parametric VaR')

subplot(4,1,2)       % add first plot in 4 x 1 grid
plot(parametric2)
title('unweighted Parametric VaR')

subplot(4,1,3)       % add second plot in 4 x 1 grid
plot(historical)       
title('Historical VaR')

subplot(4,1,4)       % add second plot in 4 x 1 grid
plot(monte)       
title('Monte Carlo VaR')

%backtest
portreturn = zeros(m-day,1);
longportloss = zeros(m-day,1);
for i = 1:m-day
    portreturn(i) = (value(i) - value(i+day))/value(i+day);
end

for i = 1:m-day
    longportloss(i) = v0 - v0*(1+portreturn(i));
end

pexception = zeros(a,1);
p2exception = zeros(a,1);
hexception = zeros(a,1);
mexception = zeros(a,1);
for i = 1:a
    if longportloss(i)>parametric(i)
       pexception(i) = 1;
    end
    if longportloss(i)>parametric2(i)
       p2exception(i) = 1;
    end
    if longportloss(i)>historical(i)
       hexception(i) = 1;
    end
    if longportloss(i)> -monte(i)
       mexception(i) = 1;
    end
end

pexceptionpy = zeros(a-252,1);
p2exceptionpy = zeros(a-252,1);
hexceptionpy = zeros(a-252,1);
mexceptionpy = zeros(a-252,1);
for i = 1:a-252
    pexceptionpy(i) = sum(pexception(i:i+252));
    p2exceptionpy(i) = sum(p2exception(i:i+252));
    hexceptionpy(i) = sum(hexception(i:i+252));
    mexceptionpy(i) = sum(mexception(i:i+252));
end

figure
subplot(2,1,1)       % add first plot in 3 x 1 grid
plot(pexceptionpy)
title('Equivalent weighted Parametric VaR Exception')

subplot(2,1,2)       % add second plot in 3 x 1 grid
plot(p2exceptionpy)       
title('unweighted Parametric VaR Exception')


xlswrite('weighted parametric VaR',parametric)
xlswrite('windowed parametric VaR',parametric2)
xlswrite('historical VaR',historical)
xlswrite('Monte Carlo VaR',monte)
xlswrite('weighted parametric exception',pexceptionpy)
xlswrite('windowed parametric exception',p2exceptionpy)
xlswrite('Acual Loss',longportloss)
