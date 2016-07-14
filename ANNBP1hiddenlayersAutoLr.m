%--------------BP-------------%
clear all;
clc;
close all;

SamNum=200;
TestSamNum=100;
HiddenUnit1Num=10;
InDim=2;
OutDim=1;
%Obtain inputs and outputs of samples
SamIn1=8*rand(1,SamNum)-4;
SamIn2=8*rand(1,SamNum)-4;
SamIn=[SamIn1;SamIn2];
SamOut=0.25*(2+sin(SamIn1)+cos(SamIn2));

TestSamIn1=-4:0.05:4;
TestSamIn2=-4:0.05:4;
TestSamIn=[TestSamIn1;TestSamIn2];
TestSamOut=0.25*(2+sin(TestSamIn1)+cos(TestSamIn2));

MaxEpochs=10000;
E0=0.001;%target error
W1=0.2*rand(HiddenUnit1Num,InDim)-0.1;
B1=0.2*rand(HiddenUnit1Num,1)-0.1;
W2=0.2*rand(OutDim,HiddenUnit1Num)-0.1;
B2=0.2*rand(OutDim,1)-0.1;
Dw1Ex=zeros(HiddenUnit1Num,InDim);
Db1Ex=zeros(HiddenUnit1Num,1);
for i=1:MaxEpochs
    %Hidden layer and the output value of the forward propagation network%
    u=W1*SamIn+B1*ones(1,size(SamIn,2));
    Hidden1Out=1./(1+exp(-u));
    NetworkOut=W2*Hidden1Out+B2;  
   
    Error=SamOut-NetworkOut;
    SSE=0.5*sum(Error.^2);
    if SSE<E0,break,end
   
    if (mod(i,2)==0)
        pause(0.0000001)
        plot(i,SSE,'b.-')%,title('Steepest Descent');
        xlabel('epoch');ylabel('error')
        hold on 
        grid on
    end
    %Calculate error of back - propagation
    Delta2=Error;
   a=zeros(10,200);
   for m=1:10     
       a(m,:)=Delta2.*Hidden1Out(m,:);   
   end
    Dw2Ex=sum(a,2)';
    Db2Ex=sum(Delta2);
   b=zeros(10,2,200);
   for m=1:SamNum 
        Delta1=(W2'*Delta2(m)).*Hidden1Out(:,m).*(1-Hidden1Out(:,m));
         Db1Ex=Db1Ex+Delta1;
         for n=1:2
         b(:,n,m)= Delta1*(SamIn(n,m))';
         end
   end
   Dw1Ex=sum(b,3);

dertaElr2=zeros(1,SamNum);
dertaElr=zeros(1,SamNum);

 for m=1:SamNum

dertazlr=Hidden1Out(:,m).*(1-Hidden1Out(:,m)).*(Dw1Ex*SamIn(:,m)+Db1Ex);%隐层输出对lr求导
dertazlr2=(Dw1Ex*SamIn(:,m)+Db1Ex).*(dertazlr-2*Hidden1Out(:,m).*dertazlr);%隐层输出对lr求2次导
dertaElr2(m)=(Dw2Ex*Hidden1Out(:,m)+W2*dertazlr+Db2Ex)^2-Delta2(m)*sum(Dw2Ex*dertazlr+W2*dertazlr2);%E对lr求导在lr=0处的值
 dertaElr(m)=-Delta2(m)*(Dw2Ex*Hidden1Out(:,m)+W2*dertazlr+Db2Ex);%E对lr求导
end

     S=-sum(dertaElr)/sum(dertaElr2);
     W1=W1+S*Dw1Ex;
     B1=B1+S*Db1Ex;
       W2=W2+S*Dw2Ex;
       B2=B2+S*Db2Ex;
    Dw1Ex=zeros(HiddenUnit1Num,InDim);
    Db1Ex=zeros(HiddenUnit1Num,1);
end
u=W1*TestSamIn+B1*ones(1,size(TestSamIn,2));
Hidden1Out=1./(1+exp(-u));
TestNetworkOut=W2*Hidden1Out+B2*ones(size(TestSamOut));
k=1:SamNum;
figure(2)
plot(k,SamOut,'-r*',k,NetworkOut,'-o');
legend('Original sample output','BP output');
xlabel('Sample number');

figure(3)
plot(TestSamIn1,TestNetworkOut,'-r*',TestSamIn1,TestSamOut,'-o');
legend('Test sample output','BP output');
xlabel('Sample');


