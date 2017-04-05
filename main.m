function [  ] = main(  )

data_obj=tdfread('Training_Data.txt','tab');
data_cell=struct2cell(data_obj);
data_mat=cell2mat(data_cell');
X=data_mat(:,2:71);
Y=data_mat(:,72);

% for i=1:3
%     exhaustive_search( X,Y, i );
% end

exhaustive_search( X,Y, 3 );

% sfs_search( X,Y );

% SFS, 1-5 Features
load sfs
sfs.nn.resub
sfs.nn.loo
sfs.dlda.resub
sfs.dlda.loo
sfs.svm.resub
sfs.svm.loo

% Exhaustive, 1 Feature
load exhasutive1
exhasutive1=exhasutive;
[m,k]=min(exhasutive1.nn.resub(:,2))
feature_set(1)=exhasutive1.nn.resub(k,1)

[m,k]=min(exhasutive1.nn.loo(:,2))
feature_set(2)=exhasutive1.nn.resub(k,1)

[m,k]=min(exhasutive1.dlda.resub(:,2))
feature_set(3)=exhasutive1.nn.resub(k,1)

[m,k]=min(exhasutive1.dlda.loo(:,2))
feature_set(4)=exhasutive1.nn.resub(k,1)

[m,k]=min(exhasutive1.svm.resub(:,2))
feature_set(5)=exhasutive1.nn.resub(k,1)

[m,k]=min(exhasutive1.svm.loo(:,2))
feature_set(6)=exhasutive1.nn.resub(k,1)

% Exhaustive, 2 Features
load exhasutive2
exhasutive2=exhasutive;
[m,k]=min(exhasutive2.nn.resub(:,3))
feature_set(7)=exhasutive2.nn.resub(k,1:2)

[m,k]=min(exhasutive2.nn.loo(:,3))
feature_set(8)=exhasutive2.nn.resub(k,1:2)

[m,k]=min(exhasutive2.dlda.resub(:,3))
feature_set(9)=exhasutive2.nn.resub(k,1:2)

[m,k]=min(exhasutive2.dlda.loo(:,3))
feature_set(10)=exhasutive2.nn.resub(k,1:2)

[m,k]=min(exhasutive2.svm.resub(:,3))
feature_set(11)=exhasutive2.nn.resub(k,1:2)

[m,k]=min(exhasutive2.svm.loo(:,3))
feature_set(12)=exhasutive2.nn.resub(k,1:2)

% Exhaustive, 3 Features
load exhasutive3
exhasutive3=exhasutive;
[m,k]=min(exhasutive3.nn.resub(:,4))
feature_set(13)=exhasutive3.nn.resub(k,1:3)

[m,k]=min(exhasutive3.nn.loo(:,4))
feature_set(14)=exhasutive3.nn.resub(k,1:3)

[m,k]=min(exhasutive3.dlda.resub(:,4))
feature_set(15)=exhasutive3.nn.resub(k,1:3)

[m,k]=min(exhasutive3.dlda.loo(:,4))
feature_set(16)=exhasutive3.nn.resub(k,1:3)

[m,k]=min(exhasutive3.svm.resub(:,4))
feature_set(17)=exhasutive3.nn.resub(k,1:3)

[m,k]=min(exhasutive3.svm.loo(:,4))
feature_set(18)=exhasutive3.nn.resub(k,1:3)


% Test
data_obj=tdfread('Testing_Data.txt','tab');
data_cell=struct2cell(data_obj);
data_mat=cell2mat(data_cell');
X=data_mat(:,2:71);
Y=data_mat(:,72);


%  SFS
for i=1:5 
    %3NN, resub
    sfs.nn.resub(1:i)
    test(X(:,sfs.nn.resub(1:i)),Y, 1)
    %3NN, loo
    sfs.nn.loo(1:i)
    test(X(:,sfs.nn.loo(1:i)),Y, 1)
end
for i=1:5 
    %dlda, resub
    sfs.dlda.resub(1:i)
    test(X(:,sfs.dlda.resub(1:i)),Y, 2)
    %dlda, loo
    sfs.dlda.loo(1:i)
    test(X(:,sfs.dlda.loo(1:i)),Y, 2)
end

for i=1:5 
    %svm, resub
    sfs.svm.resub(1:i)
    test(X(:,sfs.svm.resub(1:i)),Y, 3)
    %svm, loo
    sfs.svm.loo(1:i)
    test(X(:,sfs.svm.loo(1:i)),Y, 3)
end

%exhaustive
for j=1:3
for i=[1,2,7,8,13,14]+j-1
    feature_set(i)
    test(X(:,feature_set(i)),Y, j)
end
end

