package org.lemming.interfaces;

/**
 * The basic interface for holding a localization following the ViSP standard
 * @author Ronny Sczech
 *
 */
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
