package org.lemming.pipeline;

import org.lemming.interfaces.LocalizationInterface;

public class Localization implements LocalizationInterface {

	final private double X,Y;
	final protected long ID;
	static private long curID = 0;
	private boolean isLast;
	final private long frame;
	final private double intensity;
	
	
	public Localization(double x, double y, double intensity, long frame) {
		X=x; Y=y; ID=curID++; this.frame=frame; isLast=false; this.intensity = intensity;
	}
	
	@Override
	public boolean isLast() {
		return isLast;
	}

	/**
	 * @param isLast - switch
	 */
	@Override
	public void setLast(boolean isLast) {
		this.isLast = isLast;
	}

	@Override
	public double getX() {
		return X;
	}

	@Override
	public double getY() {
		return Y;
	}

	@Override
	public long getFrame() {
		return frame;
	}
	
	@Override
	public double getIntensity() {
		return intensity;
	}

	@Override
	public String toString(){
		return "" + getX() + "\t" + getY() + "\t" + getIntensity() + "\t" + getFrame();
	}	
}
