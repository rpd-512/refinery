clear; clc;

disp('PWD =');
disp(pwd);

testdir = fullfile(pwd,'sim_test');
if ~exist(testdir,'dir')
    mkdir(testdir);
end

fid = fopen(fullfile(testdir,'test.txt'),'w');
disp(['fid = ', num2str(fid)]);

fprintf(fid,'hello\n');
fclose(fid);

disp('basic fopen OK');

% Now test openEMS writer with MINIMAL objects
FDTD = InitFDTD();
CSX  = InitCSX();

WriteOpenEMS(testdir,'testcase',FDTD,CSX);

disp('WriteOpenEMS OK');
