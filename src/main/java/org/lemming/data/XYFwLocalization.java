package org.lemming.data;

import java.util.Arrays;

/**
 * A localization with X,Y, ID, frame and a window of pixels around it.
 * 
 * @author Joe Borbely
 *
 */
public class XYFwLocalization extends XYFLocalization {

	private float[] window;
	
	/**
	 * @return Window
	 */
	public float[] getWindow() { return window; }

	/**
	 * @param window - window
	 * @param frame - frame
	 * @param x - x
	 * @param y - y
	 */
	public XYFwLocalization(float[] window, long frame, double x, double y) {
		super(frame, x, y);
		
		this.window = Arrays.copyOf(window, window.length);
	}
	
	/**
	 * @param window - window
	 * @param frame - frame
	 * @param x - x
	 * @param y - y
	 * @param ID - ID
	 */
	public XYFwLocalization(float[] window, long frame, double x, double y, long ID) {
		super(frame, x, y, ID);
		
		this.window = Arrays.copyOf(window, window.length);
	}
	
}
