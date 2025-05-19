% Code to import .csv data with shot information and convert it to a usable
% format for the processing software. 

% May 2025

clear

%%

here = '/Users/nadineigonin/Dropbox/MAC/Documents/Research/ActiveSeismic/ROC2025_Testing/SourceTesting';

cd(here)

%% AUTO IMPORT Set up the Import Options and import the data
% opts = delimitedTextImportOptions("NumVariables", 3);
% 
% % Specify range and delimiter
% opts.DataLines = [1, 5];
% opts.Delimiter = ",";
% 
% % Specify column names and types
% opts.VariableNames = ["week2365", "ms315255879", "Subms373058"];
% opts.VariableTypes = ["string", "double", "double"];
% 
% % Specify file level properties
% opts.ExtraColumnsRule = "ignore";
% opts.EmptyLineRule = "read";
% 
% % Specify variable properties
% opts = setvaropts(opts, "week2365", "WhitespaceRule", "preserve");
% opts = setvaropts(opts, "week2365", "EmptyFieldRule", "auto");
% opts = setvaropts(opts, ["ms315255879", "Subms373058"], "TrimNonNumeric", true);
% opts = setvaropts(opts, ["ms315255879", "Subms373058"], "ThousandsSeparator", ",");
% 
% % Import the data
% T = readtable("/Users/nadineigonin/Dropbox/MAC/Documents/Research/ActiveSeismic/ROC2025_Testing/SourceTesting/TB_INT00142.csv", opts);

%%
opts = delimitedTextImportOptions("NumVariables", 3);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["week2365", "ms315255879", "Subms373058"];
opts.VariableTypes = ["categorical", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "week2365", "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["ms315255879", "Subms373058"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["ms315255879", "Subms373058"], "ThousandsSeparator", ",");

% Import the data
T = readtable("/Users/nadineigonin/Dropbox/MAC/Documents/Research/ActiveSeismic/ROC2025_Testing/SourceTesting/TB_INT00142.csv", opts);


% Clear temporary variables
clear opts

%% Pull out information

ltot = size(T,1);

count = 1;

for i = 1:3:ltot

    % get the week
    temp = string(table2array(T(i,1)));
    % Use regular expression to extract the number
    number = regexp(temp, '\d+', 'match');

    % Convert the matched string to a numeric value
    week(count) = str2double(number{1});

    % Milliseconds and sub milli seconds
    ms(count) = table2array(T(i,2));
    subms(count) = table2array(T(i,3));

    % Lat long
    temp = string(table2array(T(i+1,1)));

    % Use regular expression to extract numbers
    numbers = regexp(temp, '\d+\.\d+', 'match');

    % Convert the matched strings to numeric values
    long(count,1) = str2double(numbers{1}); % First number (96.75764)
    lat(count,1) = str2double(numbers{2}); % Second number (32.98605)

    count = count + 1;

end

%% Convert to datetime

% The start date appears to be January 6, 1980
tstart = datetime(1980,1,6);

% Get the month and day from the week
for i = 1:length(week)

    week_i = week(i);
    % Convert weeks to days (1 week = 7 days)
    days = week_i * 7;

    % Calculate the target date by adding days to the start date
    target_date = tstart + days;

    ref_date(i,1) = target_date;

    % Extract year, month, and day
    year_sub(i) = target_date.Year;
    month_sub(i) = target_date.Month;
    day_sub(i) = target_date.Day;

end

% Then, get the remaining days, hours, minutes etc. 

for i = 1:length(ms)

    ms_i = ms(i);

    % Convert to seconds
    seconds_total = ms_i / 1000;

    % Calculate days
    days_sub(i) = floor(seconds_total / (24 * 60 * 60));
    seconds_rem = mod(seconds_total, 24 * 60 * 60);

    % Calculate hours
    hours_sub(i) = floor(seconds_rem / (60 * 60));
    seconds_rem = mod(seconds_rem, 60 * 60);

    % Calculate minutes
    minutes_sub(i) = floor(seconds_rem / 60);

    % Calculate seconds
    seconds_sub(i) = mod(seconds_rem, 60);

    % And milliseconds
    

end

% Combine to date time
for i = 1:length(year_sub)

    shot_date(i,1) = ref_date(i) + days_sub(i);

    hms = datetime(0,0,0,hours_sub(i),minutes_sub(i),seconds_sub(i))-datetime(0,0,0);

    tvec_str(i,1) = shot_date(i,1)+hms;

end

%% Plot results

% Map
figure(1)
clf
hold on

plot(long,lat,'k.')

xlabel('Longitude')
ylabel('Latitude')

grid minor
box on

axis equal

set(gca,'FontSize',16)
set(gcf,'Color','w')

% Times
figure(2)
clf
hold on

hh = histogram(tvec_str);
hh.BinWidth = datetime(0,0,0,0,0,3)-datetime(0,0,0);

xlabel('Time')
ylabel('Shot')

grid minor
box on

set(gca,'FontSize',16)
set(gcf,'Color','w')

%% Export

yy = year(tvec_str);
mo = month(tvec_str);
dd = day(tvec_str);
hh = hour(tvec_str);
mm = minute(tvec_str); 
ss = second(tvec_str);

TT = [yy,mo,dd,hh,mm,ss];




