function out=myhist2(xg,yg,dx,dy,mx,my,w)

%mx(1),mx(2),my minimum, maximum values
%dx pixelsize for reconstruction 

if nargin<7
    w=1+0*xg;
end

minx=floor(mx(1)/dx);
miny=floor(my(1)/dy);
maxx=ceil(mx(2)/dx);
maxy=ceil(my(2)/dy);

x=round(xg/dx);
y=round(yg/dy);



x=x-(minx)+1;y=y-(miny)+1;

maxx2=((maxx)-minx);maxy2=(maxy-miny);

if maxx2*maxy2<1e9
  out=zeros(maxx2,maxy2);
else
    errordlg([ 'reconstruction image too big > ' num2str(maxim)])
    asf
end
sr=size(out);
ig=( x>0&y>0&x<sr(1)&y<sr(2));    
fig=find(ig);


        linind=sub2ind(size(out),x(ig),y(ig));
        for k=1:length(linind)
            out(linind(k))=out(linind(k))+w(fig(k));
        end
% out(1,1)=0;
[m,mi]=max(out(:)); out(mi)=out(mi+1);
% imagesc([mx(1) mx(2)],[my(1) my(2)],out');

    
out(:,1)=[];
out(1,:)=[];


