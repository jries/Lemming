package org.lemming.pipeline;

public class FittedLocalization extends Localization{

	final private double Z;
	final private double sX;
	final private double sY;

	public FittedLocalization(long frame, double x, double y, double z, double sx, double sy) {
		super(frame, x, y);
		this.Z=z;
		this.sX = sx;
		this.sY = sy;
	}
	
	public FittedLocalization(long ID, long frame, double x, double y, double z, double sx, double sy) {
		super(ID, frame, x, y);
		this.Z=z;
		this.sX = sx;
		this.sY = sy;
	}

	public double getZ(){
		return Z;
	}
	
	public double getsX(){
		return sX;
	}
	
	public double getsY(){
		return sY;
	}
	
	@Override
	public String toString(){
		return "" + getID() + "," + getFrame() + "," + getX() + "," + getY() + "," + getZ() + "," + getsX() + "," + getsY();
	}
}
