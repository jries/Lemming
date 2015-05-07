package org.lemming.pipeline;

public class Localization implements LocalizationInterface {

	private double X,Y;
	private long ID;
	static private long curID = 0;
	private boolean isLast = false;
	private long frame = 0;
	
	public Localization(long ID, long frame,  double x, double y) {
		this.X=x; this.Y=y; this.ID=ID; this.frame=frame;
	}
	
	public Localization(long frame, double x, double y) {
		X=x; Y=y; ID=curID++; this.frame=frame;
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
		this.isLast=isLast;
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

	@Override
	public long getFrame() {
		return frame;
	}

}
