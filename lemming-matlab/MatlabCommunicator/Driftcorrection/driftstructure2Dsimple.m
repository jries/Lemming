function [dxt,dyt]=driftstructure2Dsimple(frame, x,y)
% unit: x, y, can be pixel or nm or anyhting
%frame starts with 1, ascending order
% parameters and typical values (please adopt)
par.pixrec=.1; %pixelsize of reconstructed images in units
par.window=20; %size of region in pixels which gets fittd to determine
% displacement
par.numtimepoints=10; %number of time points evaluated 
par.maxdrift=10; %maximal drift in nm (not crucial, rather choose to
% high

%other functions needed:
%myhist2
%my2DGaussfit

%copyright: Jonas Ries, EMBL, jonas.ries@embl.de
lf=length(frame);
positions=zeros(lf,3);
positions(:,1)=frame;positions(:,2)=x;positions(:,3)=y;
% positions=[frame' x' y'];
numframes=double(positions(end,1));
%% calculate movie and FFT of movie
pixrec=par.pixrec; %in units
window=ceil((par.window-1)/2);
timepoints=par.numtimepoints; %how many timepoints
maxdrift=par.maxdrift; %in units



mx=[min(min(positions(:,2))-maxdrift) max(max(positions(:,2))+maxdrift)]; %ROI which is used for drift correction. 
my=[min(min(positions(:,3))-maxdrift) max(max(positions(:,3))+maxdrift)]; %You can put your own routine here


dummypic=myhist2(1,1,pixrec,pixrec,mx,my); %determine size of reconstructed image
srec=size(dummypic);

nfftexp=2^ceil(log2(max(srec))); %for fft use power of 2
noff=nfftexp/2+1; 
disp('make movie')
Fmovier=makemovie;  %calculate fourier transforms of reconstructed images
disp('find displacement')
[ddx, ddy]= finddisplacements2; %determine displacements

ddx=ddx*pixrec; %convert displacements into nm
ddy=ddy*pixrec;

%% bin displacements
[dx,sdx]=bindisplacementfit(ddx); %determine displacement for each time point
[dy,sdy]=bindisplacementfit(ddy);


s=size(ddx);
ddxplot=ddx;
ddyplot=ddy;
for kn=1:s(1)
    ddxplot(:,kn)=ddx(:,kn)-ddx(kn,kn)+dx(kn);
    ddyplot(:,kn)=ddy(:,kn)-ddy(kn,kn)+dy(kn);
end
sdx=std(ddxplot,1,2); %std for each time point, used for interpolation
sdy=std(ddyplot,1,2);


%plot results
figure(22)
subplot(2,2,1)
hold off
plot(ddxplot)
hold on
plot(dx,'k','LineWidth',1.5);
plot(sdx,'k:')
subplot(2,2,2)
hold off
plot(ddyplot)
hold on
plot(dy,'k','LineWidth',1.5);
plot(sdy,'k:')


%interpolate displacemnt for all frames
cfit1=(0:length(dx)-1)*binframes+binframes/2+1; %positions of time points
ctrue=1:numframes; %positions of frames
dxt = csaps(cfit1,double(dx),[],ctrue,1./sdx.^2) ;
dyt = csaps(cfit1,double(dy),[],ctrue,1./sdy.^2) ;

%
figure(22)
subplot(2,1,2)
hold off
plot(cfit1,dx,'o',ctrue,dxt,'k')
hold on
plot(cfit1,dy,'o',ctrue,dyt,'r')
xlabel('frame')
ylabel('dx, dy (units of x,y)')

% fitposc=adddrift(positions,dxt,dyt); %recalculate positions

function Fmovier=makemovie %calculate fourier transforms of images
binframes=floor(numframes/timepoints);
indold=1;
Fmovier=zeros(nfftexp,nfftexp,timepoints,'single');
for k=1:timepoints
    indnew=find(positions(:,1)>=k*binframes,1,'first');
    posframe=positions(indold:indnew,:);
    indold=indnew+1;
    image=myhist2(posframe(:,2),posframe(:,3),pixrec,pixrec,mx,my); %reconstruct single image 
    Fmovier(:,:,k)=fft2(image,nfftexp,nfftexp);
end
end

function [ddx, ddy]= finddisplacements2 % find displacements
s=size(Fmovier);
dnumframesh =s(3);
ddx=zeros(dnumframesh);ddy=zeros(dnumframesh);

for k=1:dnumframesh-1
    disp(k/dnumframesh)
    for l=k+1:dnumframesh
        cc=Fmovier(:,:,k).*conj(Fmovier(:,:,l));
        ccf=fftshift(ifft2(cc));
        [mx,my]=findmaximumgauss(real(ccf),window); %maximum by Gaussian fitting
        dxh=mx-noff; dyh=my-noff;
        ddx(k,l)=dxh; ddy(k,l)=dyh;
        ddx(l,k)=-dxh; ddy(l,k)=-dyh;
    end
end
end

function [x,y]=findmaximumgauss(img,window)
s=size(img);
win=maxdrift/pixrec; %maxdrift
cent=round(s(1)/2-win:s(1)/2+win);
imfm=img(cent,cent);
imfm=filter2(ones(5)/5^2,imfm); %filter a little for better maximum search
[inten,ind]=max(imfm(:)); %determine pixel with maximum intensity to center roi for fitting
[mxh,myh]=ind2sub(size(imfm),ind);
mxh=mxh+cent(1)-1;
myh=myh+cent(1)-1;
%now determine maximum
smallframe=double(img(mxh-window:mxh+window,myh-window:myh+window));
fitout=my2Dgaussfit(smallframe,[window+1,window+1,inten,min(smallframe(:)),max(2,3/window),max(2,3/window),0],3,1);
x=mxh-window+fitout(1)-1;y=myh-window+fitout(2)-1;
end
end

function [dx2,sdx2]=bindisplacementfit(ddx)
sf=size(ddx);
%idea: we measure displacements between every frames (dxik=xi-xk, xi is 
%displacement for frame i). Use all xi as fit parameters, fit function
%calculates dxik. Robust fit.

fp0=zeros(sf(1)-1,1);
options=statset('nlinfit');
options=statset(options,'Robust','on');
[fp,r,J,COVB,mse] = nlinfit(ddx(:),ddx(:),@bindispf,fp0,options);
ci = nlparci(fp,r,'covar',COVB);
dx2=[0; fp];
sx=ci(:,2)-ci(:,1);
sdx2=[mean(sx);sx];
end

function out=bindispf(fp,ddx)
fph=[0; fp];
ddxf=zeros(length(fph));
for k=1:length(fph)
    ddxf(k,:)=fph(k)-fph; %calculate difference
end
out=ddxf(:);
end

function posc=adddrift(pos,posx,posy)
posc=pos;
posc(:,2)=pos(:,2)-posx(posc(:,1))';
posc(:,3)=pos(:,3)-posy(posc(:,1))';
end

