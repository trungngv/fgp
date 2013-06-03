truexy <- read.csv('motorcycle.csv')
pred <- read.csv('fmean.csv')
samples <- read.csv('samples.csv')

#par('mar', c(1,1,1,1))
par(mgp = c(2,0.5,0))
plot(samples,xlab='Time (ms)',ylab='Accelration (g)',col='blue','pch'='.',cex=1.5, axes = FALSE)
axis(1,seq(0,60,10))
axis(2,seq(-150,100,50))
points(truexy[,1],truexy[,2],col='black',pch='x')
lines(xy[,1],xy[,2],col='red',cex=10)

#