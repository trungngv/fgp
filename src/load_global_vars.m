function load_global_vars()
%LOADGLOBAL load_global_vars()
% Load 'x' and 'y' from a designated matfile and store them in two global
% variables 'globalx' and 'globaly'.
disp('load global vars called')
load('tnvglobals.mat','x','y');
global globalx;
global globaly;
globalx = x;
globaly = y;


