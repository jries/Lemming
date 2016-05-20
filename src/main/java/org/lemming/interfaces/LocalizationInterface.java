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
	Number getX();
	
	/**
	 * @return y
	 */
	Number getY();
	
	/**
	 * @return intensity
	 */
	Number getIntensity();
	
	/**
	 * @return frame
	 */
	Long getFrame();
}
