close all;
clear;
clc;

unit = 1e-3; % mm

% --- FDTD ---
FDTD = InitFDTD('EndCriteria',1e-5);
FDTD = SetGaussExcite(FDTD, 4e9, 2e9);
BC = {'PML_8','PML_8','PML_8','PML_8','PML_8','PML_8'};
FDTD = SetBoundaryCond(FDTD, BC);

% --- CSX ---
CSX = InitCSX('DeltaUnit', unit);

% Grid
mesh.x = -25:5:25;
mesh.y = -25:5:25;
mesh.z = [0 1.6 10];
CSX = DefineRectGrid(CSX, unit, mesh);

% Materials
CSX = AddMaterial(CSX, 'FR4');
CSX = SetMaterialProperty(CSX,'FR4','Epsilon',4.4);

CSX = AddMetal(CSX,'PEC');

% Substrate
CSX = AddBox(CSX,'FR4',1,[-20 -20 0],[20 20 1.6]);

% Ground
CSX = AddBox(CSX,'PEC',2,[-20 -20 0],[20 20 0]);

% Patch
CSX = AddBox(CSX,'PEC',3,[-10 -8 1.6],[10 8 1.6]);

% Port
[CSX,port] = AddLumpedPort(CSX,5,1,50,[0 -8 0],[0 -8 1.6],[0 0 1]);

% Write files
Sim_Path = fullfile(pwd, 'sim');
Sim_CSX  = 'patch';

if ~exist(Sim_Path,'dir')
    mkdir(Sim_Path);
end

WriteOpenEMS(Sim_Path, Sim_CSX, FDTD, CSX);


