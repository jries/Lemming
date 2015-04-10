package org.lemming.data;

import org.lemming.interfaces.Localization;

/**
 * A simple implementation of the Localization with the X, Y and ID members.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class XYLocalization implements Localization {

	private double X,Y;
	private long ID;

	static private long curID = 0;
	private boolean isLast = false;
	
	/**
	 * @param x - x
	 * @param y - y
	 * @param ID - ID
	 */
	public XYLocalization(double x, double y, long ID) {
		this.X=x; this.Y=y; this.ID=ID;
	}
	
	/**
	 * @param x - x
	 * @param y - y
	 */
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

	@Override
	public boolean isLast() {
		return isLast;
	}

	/**
	 * @param isLast - switch
	 */
	public void setLast(boolean isLast) {
		this.isLast=isLast;
	}
	
}
