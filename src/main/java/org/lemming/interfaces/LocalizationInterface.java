package org.lemming.interfaces;


public interface LocalizationInterface extends Element {

	/**
	 * @return x
	 */
	public Number getX();
	
	/**
	 * @return y
	 */
	public Number getY();
	
	/**
	 * @return intensity
	 */
	public Number getIntensity();
	
	/**
	 * @return frame
	 */
	public Long getFrame();
}
