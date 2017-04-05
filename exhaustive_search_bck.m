function [] = exhaustive_search( X,Y, feature_num )
    
    error=struct('resub',[],'loo',[]);
    exhaustive(feature_num) = struct('nn',error,'dlda',error, 'svm',error);
    
    feature_combination=combnk(1:70,feature_num);

%     resub_error=ones(size(feature_combination,1),3);
    for i=1:size(feature_combination,1)
        set=feature_combination(i,:);
        train_set=X(:,set);
        
        %3NN
        class_3nn = fitcknn(f,Y,'NumNeighbors',3);
        predicted_label=predict(class_3nn,train_set);
%         resub_error(i,1)=sum(xor(predicted_label,Y))/60;
        exhaustive(feature_num).nn.resub=[set mean(xor(predicted_label,Y))];
        
        %DiagLDA
        class_dlda=fitcdiscr(train_set,Y,'DiscrimType','diagLinear');
        predicted_label=predict(class_dlda,train_set);
%         resub_error(i,2)=sum(xor(predicted_label,Y))/60;
        exhaustive(feature_num).dlda.resub=[set mean(xor(predicted_label,Y))];

        
        %SVM
        SVMModel = fitcsvm(train_set,Y,'cost',0.5);
        predicted_label=predict(SVMModel,train_set);
%         resub_error(i,3)=sum(xor(predicted_label,Y))/60;
        exhaustive(feature_num).svm.resub=[set mean(xor(predicted_label,Y))];
                
    end
%         [max_value,index]=max(1-resub_error(:,1)); %3NN
%         exhaustive(feature_num).nn.resub= [feature_combination(index) max_value];
%             
%         [max_value,index]=max(1-resub_error(:,2)); %DiagLDA
%         exhaustive(feature_num).dlda.resub= [feature_combination(index) max_value];
%         
%         [max_value,index]=max(1-resub_error(:,3)); %3NN
%         resub_result_SVM={feature_num, feature_combination(index), max_value}
    
     a=zeros(69,70);
     
     for i=1:70
         b=[1:70];
         b(i)=[];
         a(:,i)=b;         
     end
    
    point_error_3nn=ones(size(feature_combination,1),70);
    point_error_dlda=point_error_3nn;
    point_error_svm=point_error_3nn;

    loo_error=ones(size(feature_combination,1),3);
    
    for i=1:size(feature_combination,1)
        
        for t=1:70
        %3NN
        class_3nn = fitcknn(X(a(t),set),Y(a(t)),'NumNeighbors',3);
        predicted_label=predict(class_3nn,X(t,set));
        point_error_3nn(i,t)=abs(predicted_label-Y(t));
        end
        
        exhaustive(feature_num).nn.loo=[set mean(point_error_3nn(i,:))];
        
        
        for t=1:70
        %DiagLDA
        class_dlda = fitcdiscr(X(a(t),set),Y(a(t)),'DiscrimType','diagLinear');
        predicted_label=predict(class_dlda,X(t,set));
        point_error_dlda(i,t)=abs(predicted_label-Y(t));
        end
        
        exhaustive(feature_num).dlda.loo=[set mean(point_error_dlda(i,:))];

        
        for t=1:70
        %SVM
        SVMModel = fitcsvm(X(a(t),set),Y(a(t)),'cost',0.5);
        predicted_label=predict(SVMModel,X(t,set));
        point_error_svm(i,t)=abs(predicted_label-Y(t));
        end
        
        exhaustive(feature_num).dlda.loo=[set mean(point_error_svm(i,:))];

    end
   
%       [max_value,index]=max(1-loo_error(:,1)); %3NN
%         loo_result_3NN={feature_num, feature_combination(index), max_value}
%             
%         [max_value,index]=max(1-loo_error(:,2)); %DiagLDA
%         loo_result_dlda={feature_num, feature_combination(index), max_value}
%          
%         [max_value,index]=max(1-loo_error(:,3)); %SVM
%         loo_result_SVM={feature_num, feature_combination(index), max_value}
%         
%         save loo_result_SVM loo_result_dlda loo_result_3NN loo_error resub_error resub_result_SVM resub_result_dlda resub_result_3NN 
%  
save exhaustive(feature_num)

end

