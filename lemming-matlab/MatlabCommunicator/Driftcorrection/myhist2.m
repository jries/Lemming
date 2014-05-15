function out=myhist2(xg,yg,dx,dy,mx,my)

%mx(1),mx(2),my minimum, maximum values
%dx pixelsize for reconstruction 

minx=(mx(1)/dx);
miny=(my(1)/dy);
maxx=(mx(2)/dx);
maxy=(my(2)/dy);

x=(xg/dx);
y=(yg/dy);

x=round(x-(minx)+1);y=round(y-(miny)+1);

maxx2=round(maxx-minx);maxy2=round(maxy-miny);
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
