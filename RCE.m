function [lambda_1, lambda_2] = rce_train(class1,class2,eps,lambda_max)
 %Find number of train patterns (colums)
    n_c1p = size(class1,2);
    n_c2p = size(class2,2);
	
	for i=1:n_c1p
			x_hat = min(sqrt(sum((class2-class1(:,i)*ones(1,n_c1p)).^2)));
			lambda_1(i) = min(x_hat - eps, lambda_max);
	end
	for i=1:n_c2p
			x_hat = min(sqrt(sum((class1-class2(:,i)*ones(1,n_c2p)).^2)));
			lambda_2(i) = min(x_hat - eps, lambda_max);
	end
end

function [cl] = rce_classify(class1,lambda_1,class2,lambda_2,test_patterns)
		%Test Patterns in form: num_features x num_patterns
		ind1 = []; ind2 = [];
		 %Find number of train patterns (colums)
		n_c1p = size(class1,2);
		n_c2p = size(class2,2);
		num_test_patterns = size(test_patterns,2);
		for i = 1:num_test_patterns
			test_x = test_patterns(:,i);
			dist1 = test_x*ones(1,n_c1p)-class1;
			dist1 = sqrt(diag(dist1'*dist1))';
			
			dist2 = test_x*ones(1,n_c2p)-class2;
			dist2 = sqrt(diag(dist2'*dist2))';
			
			ind1 = find(dist1 < lambda_1);
			ind2 = find(dist2 < lambda_2);
			p = 3;
			if ~isempty(ind1)
				p = 1;
			end
			if ~isempty(ind2)
				p = 2;
			end
			if (~isempty(ind1) && ~isempty(ind2))
				p = 3;
			end
			cl(i) = p;
		end
end

function [code]	= code_data(data,feats)
	class = data(:,7);
	data=data(:,1:6);
	code = zeros(size(data,1),33);
	mcv = [95 90 85 64];
	alkphos = [90 80 75 70 65 60 55 50 22];
	rest = [70 50 30 20 3
		45 35 30 25 4
		70 35 20 15 4
		4.5 3 1.5 0.5 -0.1];
	for i = 1:length(mcv)
		indx = find(data(:,1) > mcv(i));
		code(indx,i) = 1;
		data(indx,1) = 0;
	end
	for i = 1:length(alkphos)
		indx = find(data(:,2) > alkphos(i));
		code(indx,i+4) = 1;
		data(indx,2) = 0;
	end
	for j = 1:4
		for i = 1:5
			indx = find(data(:,j+2) > rest(j,i));
			code(indx,8+(j*5)+i) = 1;
			data(indx,j+2) = -1;
		end
	end
	code(:,end+1)=class;
	code = sortrows(code,7);
	code = code(:,1:33);
	code = code(:,feats);
	a = (code == 0);
	a = -1*a;
	code = code+a;
end

function [data] = prepare_uncoded(data,feats)
	data = sortrows(data,7);
	data = data(:,1:6);
	data = data(:,feats);
end

function [class1 class2] = … training_tune(c1,c2,z1,z2,test_class1,test_class2,lambda_max,eps,eta)
v = [ 1 1 1 1; 1 1 1 -1; 1 1 -1 1; 1 1 -1 -1
    1 -1 1 1; 1 -1 1 -1; 1 -1 -1 1; 1 -1 -1 -1
    -1 1 1 1; -1 1 1 -1; -1 1 -1 1; -1 1 -1 -1
    -1 -1 1 1; -1 -1 1 -1; -1 -1 -1 1; -1 -1 -1 -1];
v = v.*eta;
x = [];
	% Find greatest weight match
	for cnt=1:16
		% Load Training data
		class1 = c1;
		class2 = c2;
		% Try different descent approaches
		class1(:,z1) = class1(:,z1)+v(cnt,1)*test_class1(:,z1);
		class1(:,z2) = class1(:,z2)+v(cnt,2)*test_class1(:,z2);
		class2(:,z1) = class2(:,z1)+v(cnt,3)*test_class2(:,z1);
		class2(:,z2) = class2(:,z2)+v(cnt,4)*test_class2(:,z2);
			
		%Train
		[lambda_1 lambda_2] = rce_train(class1,class2,eps,lambda_max);

		% Classify test_class1
		cl1 = rce_classify(class1,lambda_1,class2,lambda_2,test_class1);
		% Classify test_class2
		test_patterns = test_class2; %num_features x num_patterns
			num_test_patterns = size(test_patterns,2);
		% Classify test_class2
		cl2 = rce_classify(class1,lambda_1,class2,lambda_2,test_class2);
		% Store error for this descent
		% Count #Class1 error #Class1 Ambig. #Class2 error #Class2 ambig.
		error(cnt,:) = [cnt sum(cl1==2) sum(cl1==3) sum(cl2==1) sum(cl2==3)];
	end

	% Select greatest descent
	e = (error(1,2:end)'*ones(1,15))';
	t = e-error(2:end,2:end);
	[index vals] = find(sum(t,2) ~= 0);
	[m n ] = min(sum(t(index,:),2));
	if ~isempty(index)
		if ~isempty(n)
			x = index(n);
		end
	end

	if ~isempty(x)
		vv = v(x,:);
	end
	if isempty(x)
		vv = v(7,:);
	end
	%Reload original training data 
	class1 = c1;
	class2 = c2;
	% Modify training data weights
	class1(:,z1) = class1(:,z1)+vv(1)*test_class1(:,z1);
	class1(:,z2) = class1(:,z2)+vv(2)*test_class1(:,z2);
	class2(:,z1) = class2(:,z1)+vv(3)*test_class2(:,z1);
	class2(:,z2) = class2(:,z2)+vv(4)*test_class2(:,z2);

end

% Script

% Note, load data from
% ftp://ftp.ics.uci.edu/pub/machine-learning-databases/liver-disorders/bupa.data
% name dataset matrix 'data'

% For uncoded data
% 	3 features
% 		feats = [2 5 6];
% 	4 features
		feats = [2 3 5 6];
% 	5 features
% 		feats = [2 3 4 5 6];
% 	All
% 	feats = [1:6];

% For coded data
% 	5 features
% 		feats = [19 20 21 23 29];
% 	10 features
% 		feats= [4 9 10 14 16 18 25 28 29 30];
% 	15 features
% 		feats = [1 2 3 4 7 9 10 16 19 21 25 28 29 31];
% 	All
% 		feats = [1:33];

% Uncomment below to use uncoded data 
data = prepare_uncoded(data,feats);

% Uncomment below to code data
% data = code_data(data,feats);

test_class1 = data(1:72,:)';
class1 = data(73:144,:)';
class2 = data(145:216,:)';
test_class2 = data(217:288,:)';

% The learning rates under investigation
scales=[ .05 .1 .15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95];
% Main loop
for loop=1:length(scales)
%Clear the variables
	ee = [];
	class1 = [];
	class2 = [];
	test_class1 = [];
	test_class2 = [];
	error = [];
	a = [];
	b =[];
	lambda1 = [];
	lambda2 = [];
	cl1 = [];
	cl2 = [];
	x=[];
%Re-load Data
	bupa_data;
	
	% Set Globals
	eps=1e-15;
	lambda_max=75;
	n_c1p = size(class1,2); %Number of train patterns (colums)
	n_c2p = size(class2,2); %Number of train patterns (colums)
	
	%Find the initial case
	%Train
	[lambda_1 lambda_2] = rce_train(class1,class2,eps,lambda_max);
	% Classify
	cl1 = rce_classify(class1,lambda_1,class2,lambda_2,test_class1);
	cl2 = rce_classify(class1,lambda_1,class2,lambda_2,test_class2);

% Iterate to find the best error reduction for this eta
	for count=1:10
		%Find ambiguous
		z1 = intersect(find(cl1==2),find(cl2==3));
		z2 = intersect(find(cl1==3),find(cl2==1));
		c1 = class1;c2 = class2;
		% Tune to greatest descent 
		[class1 class2] = ...
		training_tune(c1,c2,z1,z2,test_class1,test_class2,lambda_max,eps,scales(loop));
		% Classify based on tuned training samples
		cl1 = rce_classify(class1,lambda_1,class2,lambda_2,test_class1);
		cl2 = rce_classify(class1,lambda_1,class2,lambda_2,test_class2);
		% Populate the error matrix for this iteration
		ee(count,:) = [count sum(cl1==2) sum(cl1==3) sum(cl2==1) sum(cl2==3)];
	end

%Find the greatest reduction for this eta,output to stdout
	ee = ee(:,2:end);
	[b a] = min(sum(ee,2));
[ee(a,:) sum(ee(a,:),2) round((ee(a,1) + ee(a,3))/1.44) round((ee(a,2) + ee(a,4))/1.44)]
end