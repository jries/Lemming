package org.lemming.interfaces;


public interface LocalizationInterface extends Element {

	/**
	 * @return x
	 */
	public double getX();
	
	/**
	 * @return y
	 */
	public double getY();
	
	/**
	 * @return intensity
	 */
	public double getIntensity();
	
	/**
	 * @return frame
	 */
	public long getFrame();
}
