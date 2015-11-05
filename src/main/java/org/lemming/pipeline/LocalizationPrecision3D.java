package org.lemming.pipeline;

public class LocalizationPrecision3D extends Localization{

	final private double Z;
	final private double sX;
	final private double sY;
	final private double sZ;

	public LocalizationPrecision3D(double x, double y, double z, double sx, double sy, double sz, double intensity, long frame) {
		super(x, y, intensity, frame);
		this.Z = z;
		this.sX = sx;
		this.sY = sy;
		this.sZ = sz;
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
	
	public double getsZ(){
		return sZ;
	}
	
	@Override
	public String toString(){
		return "" + getX() + "\t" + getY() + "\t" + getZ() + "\t" + getsX() + "\t" + getsY() + "\t" + getsZ() + "\t" + getIntensity() +"\t" + getFrame();
	}
}
