function [] = exhaustive_search( X,Y, feature_num )
    tic
    error=struct('resub',[],'loo',[]);
    exhaustive = struct('nn',error,'dlda',error, 'svm',error);
    
    feature_combination=combnk(1:70,feature_num);
    
    for i=1:size(feature_combination,1)
%     for i=1:3

        set=feature_combination(i,:);
        train_set=X(:,set);
        
        %3NN
        class_nn = fitcknn(train_set,Y,'NumNeighbors',3);
        predicted_label=predict(class_nn,train_set);

        exhaustive.nn.resub=[exhaustive.nn.resub; [set mean(xor(predicted_label,Y))]];
        
        %DiagLDA
        class_dlda=fitcdiscr(train_set,Y,'DiscrimType','diagLinear','prior','uniform');
        predicted_label=predict(class_dlda,train_set);

        exhaustive.dlda.resub=[exhaustive.dlda.resub; [set mean(xor(predicted_label,Y))]];

        
        %SVM
        SVMModel = fitcsvm(train_set,Y,'cost',[0,0.5;0.5,0]);
        predicted_label=predict(SVMModel,train_set);

        exhaustive.svm.resub=[exhaustive.svm.resub; [set mean(xor(predicted_label,Y))]];
                
    end

    
     n=60; % n sample numbers
     a=zeros(n-1,n);     
     for i=1:n
         b=[1:n];
         b(i)=[];
         a(:,i)=b;         
     end
    
%     point_error_3nn=ones(size(feature_combination,1),70);
%     point_error_dlda=point_error_3nn;
%     point_error_svm=point_error_3nn;
      
%     point_error_3nn=ones(1,n);

    
    for i=1:size(feature_combination,1)
            point_error_3nn=ones(1,n);
            point_error_dlda=ones(1,n);
            point_error_svm=ones(1,n);
%     for i=1:3
        set=feature_combination(i,:);

        for t=1:n  %t corresponds to sample indice
        %3NN        
        class_nn = fitcknn(X(a(:,t),set),Y(a(:,t)),'NumNeighbors',3);
        predicted_label=predict(class_nn,X(t,set));
%         point_error_3nn(i,t)=abs(predicted_label-Y(t));
        point_error_3nn(t)=abs(predicted_label-Y(t));

        end
        
%         exhaustive.nn.loo=[exhaustive.nn.loo; [set mean(point_error_3nn(i,:))]];
         exhaustive.nn.loo=[exhaustive.nn.loo; [set mean(point_error_3nn)]];
        
        for t=1:n
        %DiagLDA
        class_dlda = fitcdiscr(X(a(:,t),set),Y(a(:,t)),'DiscrimType','diagLinear','prior','uniform');
        predicted_label=predict(class_dlda,X(t,set));
%         point_error_dlda(i,t)=abs(predicted_label-Y(t));
        point_error_dlda(t)=abs(predicted_label-Y(t));

        end
        
%         exhaustive.dlda.loo=[exhaustive.dlda.loo; [set mean(point_error_dlda(i,:))]];
        exhaustive.dlda.loo=[exhaustive.dlda.loo; [set mean(point_error_dlda)]];


        
        for t=1:n
        %SVM
        SVMModel = fitcsvm(X(a(:,t),set),Y(a(:,t)),'cost',[0,0.5;0.5,0]);
        predicted_label=predict(SVMModel,X(t,set));
        point_error_svm(t)=abs(predicted_label-Y(t));
        end
        
        exhaustive.svm.loo=[exhaustive.svm.loo; [set mean(point_error_svm)]];

    end
   
save (['exhaustive' num2str(feature_num) '.mat'],'exhaustive')
toc
end

