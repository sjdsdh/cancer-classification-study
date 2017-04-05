function [ true_error ] = test( test_set,Y, classifier )

        

        switch classifier
            case 1
        %3NN
        class_nn = fitcknn(test_set,Y,'NumNeighbors',3);
        predicted_label=predict(class_nn,test_set);

         true_error=mean(xor(predicted_label,Y))
         
            case 2
        %DiagLDA
        class_dlda=fitcdiscr(test_set,Y,'DiscrimType','diagLinear','prior','uniform');
        predicted_label=predict(class_dlda,test_set);

        true_error=mean(xor(predicted_label,Y))

            case 3
        %SVM
        SVMModel = fitcsvm(test_set,Y,'cost',[0,0.5;0.5,0]);
        predicted_label=predict(SVMModel,test_set);

        true_error=mean(xor(predicted_label,Y))

        end

end

