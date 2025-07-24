function reduced_SPM = reduce_SPM_filesize(SPM, savefile)

% this strips of all of the fields that are HUGE and not necessary to pass
% along data (e.g. the filenames + fullpath of the input files etc)
%
% USAGE
% reduce_SPM_filesize()
%   loads the SPM.mat of the current folder and saves the stripped version
%   to SPM_reduced.mat (recommended)
%
% reduced_SPM = reduce_SPM_filesize()
%   also returns the reduced SPM struct
% 
% reduce_SPM_filesize(SPM)
%   saves the reduced version of the provided SPM struct (no file details
%   of the original file saved then)
% 
% reduced_SPM = reduce_SPM_filesize(SPM, 0)
%   does not save the file, only returns the stripped version.
% 
% reduce_SPM_filesize(SPM, filename)
%   saves the reduced SPM.mat to filename
%
%
% DETAILED DESCRIPTION
% The following description comes from a SPM file that was 89MB before file
% reduction
% 
% Name      Size                Bytes  Class     Attributes
% 
%   SPM       1x1             127389815  struct   
%
% The only fields that we really need at the moment are the regressor names
% (to create a design from SPM). These are in SPM.xX.name
%
% This is tiny:
%
%   Name      Size            Bytes  Class    Attributes
%
%   name      1x21             1848  cell  
%
% So we simply save only this field and save a lots of space.
%
% Kai, 2012-03-09

%% load SPM.mat if not provided
if ~exist('SPM', 'var')
    load SPM.mat
    reduced_SPM.orgfile.pwd = pwd;
    reduced_SPM.orgfile.dir = dir('SPM.mat');
end

%% extract infos
reduced_SPM.xX.name = SPM.xX.name;
reduced_SPM.info = 'This is a very much reduced version of the original SPM.mat file. It only contains what is important for The Decoding Toolbox. The original details (filesize) are saved in .org_whos'
reduced_SPM.org_whos = whos('SPM');
reduced_SPM.reduced_createdate = datestr(now);

%% save file

if ~exist('savefile', 'var')
    savefile = 1; % by default, save the file
end

if savefile
    SPM = reduced_SPM;
    
    if ischar(savefile)
        filename = savefile;
    else
        filename = 'reduced_SPM.mat';
    end
    
    save(filename, 'SPM')
end