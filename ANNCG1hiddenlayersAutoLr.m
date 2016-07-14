%--------------CG-------------%
clear all;
clc;
close all;

SamNum=200;%number of training data
TestSamNum=100;%number of test data
HiddenUnit1Num=10;%number of hidden nodes
InDim=2;%number of inputs
OutDim=1;%number of output
%get training data
SamIn1=8*rand(1,SamNum)-4;%generate a vector of input1
SamIn2=8*rand(1,SamNum)-4;%generate a vector of input2
SamIn=[SamIn1;SamIn2];
SamOut=0.25*(2+sin(SamIn1)+cos(SamIn2));

TestSamIn1=-4:0.05:4;%test data
TestSamIn2=-4:0.05:4;%test data
TestSamIn=[TestSamIn1;TestSamIn2];
TestSamOut=0.25*(2+sin(TestSamIn1)+cos(TestSamIn2));%Target Output of Test Data  
MaxEpochs=10000;%maximum epochs
E0=0.0001;%target error
W1=0.2*rand(HiddenUnit1Num,InDim)-0.1;%generate weights randomly
B1=0.2*rand(HiddenUnit1Num,1)-0.1;
W2=0.2*rand(OutDim,HiddenUnit1Num)-0.1;
B2=0.2*rand(OutDim,1)-0.1;
tempg1=zeros(10,2,SamNum,3);%negative gradient
tempg2=zeros(1,10,SamNum,3);%negative gradient
tempd1=zeros(10,2,2);%search direction
tempd2=zeros(1,10,2);%search direction
tempgb1=zeros(10,1,SamNum,3);%negative gradient
tempgb2=zeros(1,SamNum,3);%negative gradient
tempdb1=zeros(10,1,2);%search direction
tempdb2=zeros(1,2);%search direction

Dw1Ex=zeros(HiddenUnit1Num,InDim,SamNum);
Db1Ex=zeros(HiddenUnit1Num,1,SamNum);
Dw2Ex=zeros(OutDim,HiddenUnit1Num,SamNum);
Db2Ex=zeros(OutDim,SamNum);
for i=1:MaxEpochs
    %Hidden layer and the output value of the forward propagation network%
    u=W1*SamIn+B1*ones(1,size(SamIn,2));
    Hidden1Out=1./(1+exp(-u));
    NetworkOut=W2*Hidden1Out+B2;   %network output
    %Determine whether to stop learning
    Error=SamOut-NetworkOut;
    SSE=sum(Error.^2);%sum of errors
    if SSE<E0,break,end
    %plot error curve
    if (mod(i,2)==0)
        pause(0.0000001)
        plot(i,SSE,'b.-')%,title('Conjugate Gradient');
        xlabel('epoch');ylabel('error')
        hold on 
        grid on
    end
   
    %%
    %%Calculate error of back - propagation
    
    Delta2=Error;
   a=zeros(HiddenUnit1Num,SamNum);
   for m=1:HiddenUnit1Num     
       a(m,:)=Delta2.*Hidden1Out(m,:);
   end
    Dw2Ex(1,:,:)=a;
    Db2Ex=Delta2;
    b=zeros(HiddenUnit1Num,InDim,SamNum);
    c=zeros(HiddenUnit1Num,1,SamNum);
    for m=1:SamNum 
        Delta1=(W2'*Delta2(m)).*Hidden1Out(:,m).*(1-Hidden1Out(:,m));
         c(:,:,m)=Delta1;
         for n=1:2
         b(:,n,m)= Delta1*(SamIn(n,m))';
         end
   end
   Dw1Ex=b;
   Db1Ex=c;
   
        tempg2(:,:,:,1)=-Dw2Ex;
        tempgb2(:,:,1)=-Db2Ex; 
        tempg1(:,:,:,1)=-Dw1Ex;       
        tempgb1(:,:,:,1)=-Db1Ex; 
%%
%     update weights
    if (i==1) %first epoch
    tempd1(:,:,1)=-sum(tempg1(:,:,:,1),3);
    tempd2(:,:,1)=-sum(tempg2(:,:,:,1),3);
    tempdb1(:,:,1)=-sum(tempgb1(:,:,:,1),3);
    tempdb2(:,1)=-sum(tempgb2(:,:,1),2);
    tempg1(:,:,:,2:end)=tempg1(:,:,:,1:end-1);
     tempg2(:,:,:,2:end)=tempg2(:,:,:,1:end-1);
     tempd1(:,:,2:end)=tempd1(:,:,1:end-1);
     tempd2(:,:,2:end)=tempd2(:,:,1:end-1);
     tempgb1(:,:,:,2:end)=tempgb1(:,:,:,1:end-1);
     tempgb2(:,:,2:end)=tempgb2(:,:,1:end-1);
     tempdb1(:,:,2:end)=tempdb1(:,:,1:end-1);
     tempdb2(:,2:end)=tempdb2(:,1:end-1);
    else 

        ga=sum(sum(sum(tempg1(:,:,:,1).^2),2))+sum(sum(sum(tempgb1(:,:,:,1).^2),2))+sum(sum(sum(tempg2(:,:,:,1).^2),2))+sum(sum(tempgb2(:,:,1).^2));
        gb=sum(sum(sum(tempg1(:,:,:,2).^2),2))+sum(sum(sum(tempgb1(:,:,:,2).^2),2))+sum(sum(sum(tempg2(:,:,:,2).^2),2))+sum(sum(tempgb2(:,:,2).^2));
        beit1=ga/gb;
        for n=1:InDim  
            for m=1:HiddenUnit1Num
        tempd1(m,n,1)=-sum(tempg1(m,n,:,1))+beit1*tempd1(m,n,2);
        tempdb1(m,1,1)=-sum(tempgb1(m,1,:,1))+beit1*tempdb1(m,1,2);
        tempd2(1,m,1)=-sum(tempg2(1,m,:,1))+beit1*tempd2(1,m,2);
            end               
        end
            tempdb2(1,1)=-sum(tempgb2(1,:,1))+beit1*tempdb2(1,2);
    end
    %%
    %line minimization
    dertaElr2=zeros(1,SamNum);
    dertaElr=zeros(1,SamNum);    
    for m=1:SamNum
    dertazlr=Hidden1Out(:,m).*(1-Hidden1Out(:,m)).*(tempd1(:,:,1)*SamIn(:,m)+tempdb1(:,:,1));
    dertazlr2=(tempd1(:,:,1)*SamIn(:,m)+tempdb1(:,:,1)).*(dertazlr-2*Hidden1Out(:,m).*dertazlr);
    dertaElr2(m)=(tempd2(:,:,1)*Hidden1Out(:,m)+W2*dertazlr+tempdb2(:,1))^2-Delta2(m)*sum(tempd2(:,:,1)*dertazlr+W2*dertazlr2);
    dertaElr(m)=-Delta2(m)*(tempd2(:,:,1)*Hidden1Out(:,m)+W2*dertazlr+tempdb2(:,1));
    end
     S=-sum(dertaElr)/sum(dertaElr2);
     %update weights
       W1=W1+S*tempd1(:,:,1);
      B1=B1+S*tempdb1(:,:,1);
      W2=W2+S*tempd2(:,:,1);
      B2=B2+S*tempdb2(:,1);
      
     tempg1(:,:,:,2:end)=tempg1(:,:,:,1:end-1);
     tempg2(:,:,:,2:end)=tempg2(:,:,:,1:end-1);
     tempd1(:,:,2:end)=tempd1(:,:,1:end-1);
     tempd2(:,:,2:end)=tempd2(:,:,1:end-1);
     tempgb1(:,:,:,2:end)=tempgb1(:,:,:,1:end-1);
     tempgb2(:,:,2:end)=tempgb2(:,:,1:end-1);
     tempdb1(:,:,2:end)=tempdb1(:,:,1:end-1);
     tempdb2(:,2:end)=tempdb2(:,1:end-1);
end
u=W1*TestSamIn+B1*ones(1,size(TestSamIn,2));
Hidden1Out=1./(1+exp(-u));
TestNetworkOut=W2*Hidden1Out+B2*ones(size(TestSamOut));
k=1:SamNum;
figure(2)
plot(k,SamOut,'-r*',k,NetworkOut,'-o');
legend('Original sample output','CG output');
xlabel('Sample number');

figure(3)
plot(TestSamIn1,TestNetworkOut,'-r*',TestSamIn1,TestSamOut,'-o');
legend('Test sample output','CG output');
xlabel('Sample');


