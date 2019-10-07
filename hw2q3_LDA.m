function hw2q2_LDA(N,mu1,mu2,sigma1,sigma2,Prior1,Prior2)
    %%%%generate data
    
    N1 = N*Prior1;
    N2 = N*Prior2;
    A1 = mvnrnd(mu1,sigma1,N1);
    A1(:,3) = 1;   %%% assign class 1
    A2 = mvnrnd(mu2,sigma2,N2);
    A2(:,3) = 2;   %%% assign class 2
    A = [A1;A2];   %%% combina the sampled data set
   
   %%%%%LDA process  
    LDAmu1 = mean(A1(:,1:2));
    LDAmu2 = mean(A2(:,1:2));
    s1 = cov(A1(:,1:2));
    s2 = cov(A2(:,1:2));
    sw = Prior1*s1+Prior2*s2;
    sb = (LDAmu1-LDAmu2)'*(LDAmu1-LDAmu2);
    invsw = (sw)^-1;
    invsw_by_sb = invsw*sb;
    [V,D] = eig(invsw_by_sb);
    if D(1,1) >D(2,2)
        w = V(:,1);
    else
        w = V(:,2);
    end
    if w(1) <0 && w(2)<0
        w = w*-1;
    end
    A(:,5) = A(:,1:2)*w;
    
    y1 = A1(:,1:2) *w;
    y2 = A2(:,1:2) *w;
    figure(1)
    y = -10:0.05:10;
    y1_mu = mean(y1);
    y1_sigma = std(y1);
    y1_pdf = mvnpdf(y',y1_mu,y1_sigma);
    y2_mu = mean(y2);
    y2_sigma = std(y2);
    y2_pdf = mvnpdf(y',y2_mu,y2_sigma);
    plot(y,y1_pdf,y,y2_pdf);
    
    xlabel('LDA direction','FontSize',16);
    ylabel('prejection distribution','FontSize',16);
    title('homework2-LDA projection for two classes','FontSize',16);
    legend("class1",'class2');
    %%% define the range of LDA threshold for C
    C = -10:0.1:10;
    LDAerr = zeros(1,length(C));
    for nn = 1:length(C)
       for mm = 1:N
           yy = (A(mm,1:2)*w);
           if yy < C(nn)
               if A(mm,3) == 2
                   LDAerr(nn) =  LDAerr(nn)+1;
               end
           else
               if A(mm,3) == 1
                    LDAerr(nn) =  LDAerr(nn)+1;
               end
           end
       end
    end
    [LDAmin,I] = min(LDAerr); %%%get the mim errorate
    %%%%prediction
    for mm = 1:N
        yy = (A(mm,1:2)*w);
        if yy < C(I)
            A(mm,4) =1;
        else
            A(mm,4) =2;
        end
    end
    figure(2)
    gscatter(A(:,1),A(:,2),A(:,3),'br','x+');
    hold on
    gscatter(A(:,1),A(:,2),A(:,4),'br','oo');
    xlabel('smaple X','FontSize',16);
    ylabel('sample y','FontSize',16);
    title('homework2-LDA prediction','FontSize',16);
    legend('x for sample class1','+ for sample class2','bo for predict-class1','ro for predict-class2','FontSize',16);
   
end