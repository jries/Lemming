package org.lemming.pipeline;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.LocalizationInterface;

public class Localization implements LocalizationInterface {

	final private double X,Y;
	final private long ID;
	static private long curID = 0;
	private boolean isLast;
	final private long frame;
	
	public Localization(long ID, long frame,  double x, double y) {
		this.X=x; this.Y=y; this.ID=ID; this.frame=frame; isLast=false;
	}
	
	public Localization(long frame, double x, double y) {
		X=x; Y=y; ID=curID++; this.frame=frame; isLast=false;
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

	@Override
	public Element deepClone() {
		return new Localization(ID, frame, X, Y);
	}

}
