clear all, close all
addpath(genpath('../../../../code/gpml'));
addpath(genpath('../../../../code/project'));
addpath(genpath('../../../../code/oldgpml'));
addpath(genpath('../../../../code/figtree-0.9.3'));
addpath(genpath('../../../../code/project'));

paper = [8 6];

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

n = 20;
x = gpml_randn(0.3, n, 1);
K = feval(covfunc{:}, hyp.cov, x);
mu = feval(meanfunc{:}, hyp.mean, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);
 
set(gca, 'FontSize', 24)
grid on
plot(x, y, 'k+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
xlabel('input, x')
ylabel('output, y')
set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
grid on
saveas(gcf, 'preReg', 'png');

nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)
z = linspace(-1.9, 1.9, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

set(gca, 'FontSize', 24);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8);

hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
xlabel('input, x')
ylabel('output, y')
set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
grid on
saveas(gcf, 'postReg', 'png');

hyp.cov = log([1; 1]);
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

clf;
grid on
set(gca, 'FontSize', 24);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8);

hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
xlabel('input, x')
ylabel('output, y')
set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
grid on
saveas(gcf, 'postReg2', 'png');

covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
meanfunc = @meanZero;
hyp.mean = [];
hyp = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, x, y);

%FULL GP
if 1
  clf;
  hold on;
  grid on
  nu = 20;
  [dummy sod] = gp_sod(hyp, covfunc, likfunc, x, y, nu, 'r');
  [m s2] = gp_sod(hyp, covfunc, likfunc, x, y, sod, 'g', z);
  z = linspace(-1.9, 1.9, 101)';
  set(gca, 'FontSize', 24);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8);

  hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
  axis([-1.9 1.9 -0.9 3.9])
  xlabel('input')
  ylabel('SoD output')
  set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
  saveas(gcf, 'FullExample', 'png');
end

%------------------------------------------------------------------
% SoD
%------------------------------------------------------------------
if 1
  clf;
  hold on;
  grid on
  nu = 7;
  [dummy sod] = gp_sod(hyp, covfunc, likfunc, x, y, nu, 'r');
  [m s2] = gp_sod(hyp, covfunc, likfunc, x, y, sod, 'g', z);
  z = linspace(-1.9, 1.9, 101)';
  set(gca, 'FontSize', 24);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8);
  plot(x(sod,:), 1,'ko', 'MarkerSize', 12)

  hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
  axis([-1.9 1.9 -0.9 3.9])
  xlabel('input')
  ylabel('SoD output')
  set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
  saveas(gcf, 'SoDExample', 'png');
end
%------------------------------------------------------------------
% Local
%------------------------------------------------------------------
if 1
  [ci cc] = rrClust(x, 8);
  [m s2] = gp_local(hyp, covfunc, meanfunc, likfunc, x, y, ci, cc, z);
  clf;
  hold on;
  grid on
  z = linspace(-1.9, 1.9, 101)';
  set(gca, 'FontSize', 24);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8);

  % Plot x and ys colored by cluster.
  colors = {'k+', 'k<', 'ko', 'ks', 'kv', 'k*', 'k>'};
  clusts = unique(ci);
  for c = clusts
    plot(x(ci==c,:), y(ci==c, :), colors{mod(c, length(colors))+1}, 'MarkerSize', 12);
  end
  hold on; plot(z, m, 'k', 'LineWidth', 2); 
  axis([-1.9 1.9 -0.9 3.9])
  xlabel('input')
  ylabel('Local GP output')
  set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
  saveas(gcf, 'LocalExample', 'png');
end
%------------------------------------------------------------------
% FITC
%------------------------------------------------------------------
if 1
  u = x(sod,:);
  covfuncF = {@covFITC, {covfunc}, u};
  [mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, x, y, z);
  clf;
  set(gca, 'FontSize', 24)
  f = [mF+2*sqrt(s2F); flipdim(mF-2*sqrt(s2F),1)];
  fill([z; flipdim(z,1)], f, [7 7 7]/8)
  hold on; 
  plot(z, mF, 'Color', 'k', 'LineWidth', 2); 
  plot(x, y, 'k+', 'MarkerSize', 12)
  plot(u,1,'ko', 'MarkerSize', 12)
  grid on
  xlabel('input')
  ylabel('FITC output')
  axis([-1.9 1.9 -0.9 3.9])
  saveas(gcf, 'FITCExample', 'png');
end
%------------------------------------------------------------------
% IFGT
%------------------------------------------------------------------
if 1
  clf;
  hold on;
  grid on
  hypIFGT = [hyp.cov; hyp.lik];
  [m s2] = gpr_iter(hypIFGT, covfunc, x, y, 100, 0.1^7, z);
  z = linspace(-1.9, 1.9, 101)';
  set(gca, 'FontSize', 24);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8);

  hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
  axis([-1.9 1.9 -0.9 3.9])
  xlabel('input')
  ylabel('IFGT output')
  set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
  saveas(gcf, 'IFGTExample07', 'png');

  clf;
  [m s2] = gpr_iter(hypIFGT, covfunc, x, y, 100, 0.1^1.5, z);
  z = linspace(-1.9, 1.9, 101)';
  set(gca, 'FontSize', 24);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8);
  hold on; plot(z, m, 'k', 'LineWidth', 2); plot(x, y, 'k+', 'MarkerSize', 12)
  axis([-1.9 1.9 -0.9 3.9])
  xlabel('input')
  ylabel('IFGT output')
  set(gcf, 'PaperSize', paper, 'PaperUnits', 'centimeter');
  saveas(gcf, 'IFGTExample1', 'png');
end

