set.seed(1)

N = 10000
b0 = 2.5
b1 = -1.7
b2 = -2 
x = sort(rnorm(N))
logit_p = b0 + b1 * x + b2 * x^2
sigmoid = function(x) 1 / (1 + exp(-x))
p_true = sigmoid(logit_p)
# plot(x, p, type='l')

y = rbinom(N, size=1, p=p_true)
# plot(x, y)

dat = data.frame(x, y, p_true)
write.csv(dat, 'dat.txt', row.names=F, quote=F)
