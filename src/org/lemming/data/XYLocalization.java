package org.lemming.data;

public class XYLocalization implements Localization {

	double X,Y;
	long ID;
	
	static long curID = 0;
	
	public XYLocalization(double x, double y, long id) {
		X=x; Y=y; ID=id;
	}
	
	public XYLocalization(double x, double y) {
		X=x; Y=y; ID=curID++;
	}
	
	@Override
	public long getID() {
		return ID;
	}

	@Override
	public double getX() {
		return X;
	}

	@Override
	public double getY() {
		return Y;
	}
	
}
