package org.lemming.interfaces;


public interface LocalizationInterface extends Element {
	/**
	 * @return ID
	 */
	public long getID();
	
	/**
	 * @return x
	 */
	public double getX();
	
	/**
	 * @return y
	 */
	public double getY();	
	
	/**
	 * @return frame
	 */
	public long getFrame();
}
