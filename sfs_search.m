function [] = sfs_search( X,Y )
    tic
    error=struct('resub',[],'loo',[]);
    sfs = struct('nn',error,'dlda',error, 'svm',error);
    
%     feature_combination=[1:70];
    best_set=[];
    result=[];
    i=1;
    n=70;
    for j=1:5

    for i=1:70
%     for i=1:3
        
        train_set=X(:,[best_set i]);
        
        %3NN
        class_nn = fitcknn(train_set,Y,'NumNeighbors',3);
        predicted_label=predict(class_nn,train_set);
        e(i)=mean(xor(predicted_label,Y));
                
    end
        [v,k]=min(e)
        while ismember(k, best_set)
           e(k)=1;
           [v,k]=min(e);           
        end
%            g=e;
% 
%         while ismember(k, best_set)
%            g(k)=[];           
%            k=find(e==min(g));           
%            k=min(k)
%         end
        
        best_set=[best_set k]
        result=[result e(k)]
    end
    sfs.nn.resub= [best_set result];
      
    %DLDA-resub
    best_set=[];
    result=[];
    i=1;
    n=70;
    for j=1:5

    for i=1:70
%     for i=1:3
        
        train_set=X(:,[best_set i]);
        
        %dlda
        class_dlda=fitcdiscr(train_set,Y,'DiscrimType','diagLinear','prior','uniform');
        predicted_label=predict(class_dlda,train_set);
        e(i)=mean(xor(predicted_label,Y));
                
    end
        [v,k]=min(e)
        while ismember(k, best_set)
           e(k)=1;
           [v,k]=min(e);           
        end
       
        best_set=[best_set k]
        result=[result v]
    end
    sfs.dlda.resub= [best_set result];
    
    
    %SVM-resub
        
    best_set=[];
    result=[];
    i=1;
    n=70;
    for j=1:5

    for i=1:70
%     for i=1:3
        
        train_set=X(:,[best_set i]);
        
        %svm
        SVMModel = fitcsvm(train_set,Y,'cost',[0,0.5;0.5,0]);
        predicted_label=predict(SVMModel,train_set);
        e(i)=mean(xor(predicted_label,Y));
                
    end
        [v,k]=min(e)
        while ismember(k, best_set)
           e(k)=1;
           [v,k]=min(e);           
        end
       
        best_set=[best_set k]
        result=[result v]
    end
    sfs.svm.resub= [best_set result];

%     3NN-Loo
     n=60; % n sample numbers
     a=zeros(n-1,n);     
     for i=1:n
         b=[1:n];
         b(i)=[];
         a(:,i)=b;         
     end
    
            
    point_error_3nn=ones(1,n);

    loo_3nn=ones(1,70);
    loo_dlda=ones(1,70);
    loo_svm=ones(1,70);

    best_set=[];
    result=[];
    i=1;
    for j=1:5

    for i=1:70
    
        
        
      
        for t=1:n  %t corresponds to sample indice
%         3NN        
        class_nn = fitcknn(X(a(:,t),[best_set i]),Y(a(:,t)),'NumNeighbors',3);
        predicted_label=predict(class_nn,X(t,[best_set i]));
        point_error_3nn(t)=abs(predicted_label-Y(t));

        end
        loo_3nn(i)=mean(point_error_3nn);
    end
    
    [v,k]=min(loo_3nn)

    while ismember(k, best_set)
           loo_3nn(k)=1;           
           [v,k]=min(loo_3nn);                 
        
    end
       
        best_set=[best_set k]
        result=[result v]
    end
    sfs.nn.loo= [best_set result];
    
    
    %DLDA Loo
    point_error_dlda=ones(1,n);
    
    best_set=[];
    result=[];
    i=1;
    for j=1:5

    for i=1:70

        
        
      
        for t=1:n  %t corresponds to sample indice
        %dlda
        class_dlda = fitcdiscr(X(a(:,t),[best_set i]),Y(a(:,t)),'DiscrimType','diagLinear','prior','uniform');
        predicted_label=predict(class_dlda,X(t,[best_set i]));
        point_error_dlda(t)=abs(predicted_label-Y(t));

        end
        loo_dlda(i)=mean(point_error_dlda);
    end
    
    [v,k]=min(loo_dlda)

    while ismember(k, best_set)
           loo_dlda(k)=1;
           [v,k]=min(loo_dlda);                 
        
    end
       
        best_set=[best_set k]
        result=[result v]
    end
    sfs.dlda.loo= [best_set result];
    
    
    %SVM Loo
    point_error_svm=ones(1,n);
    
    best_set=[];
    result=[];
    i=1;
    for j=1:5

    for i=1:70

        
        
      
        for t=1:n  %t corresponds to sample indice
        %svm
         SVMModel = fitcsvm(X(a(:,t),[best_set i]),Y(a(:,t)),'cost',[0,0.5;0.5,0]);
        predicted_label=predict(SVMModel,X(t,[best_set i]));
        point_error_svm(t)=abs(predicted_label-Y(t));

        end
        loo_svm(i)=mean(point_error_svm);
    end
    
    [v,k]=min(loo_svm)

    while ismember(k, best_set)
           loo_dlda(k)=1;
           [v,k]=min(loo_svm);                 
        
    end
       
        best_set=[best_set k]
        result=[result v]
    end
    sfs.svm.loo= [best_set result];
    
    
    
   
save sfs
toc
end

